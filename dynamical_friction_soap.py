import sys, os, time
import numpy as np
import h5py
import argparse
import fnmatch
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import tracemalloc
#from HBTReader import HBTReader
from scipy.special import erf

# Min number of parts for velocity dispersion
Min_Nngb = 48

timing_lock = Lock()

def centre_periodic_positions(pos, centre, boxsize):
  '''
  Periodic wrapping of particle positions after centring
  '''

  pos = pos - centre

  for i in range(3):
    pos[pos[:,i] > 0.5 * boxsize[i], i] -= boxsize[i]
    pos[pos[:,i] < -0.5 * boxsize[i], i] += boxsize[i]

  return pos

def load_physical_data(h5group, prop, ascale, indices=None):
  '''
  Load data from hdf5 file/group and convert to physical units
  '''

  if not indices is None:

    # Get slice and then index
    minIdx = np.min(indices)
    maxIdx = np.max(indices)
    data = h5group[prop][minIdx:maxIdx+1][indices-minIdx]

  else:
    data = h5group[prop][:]

  aexp = h5group[prop].attrs['a-scale exponent'][0]
  if aexp != 0:
    data *= ascale**aexp

  return data

def eccentricity(do_stars, pos, vel, const_G, rsort, menc, potential_outer):
  '''
  Calculate eccentricity parameter epsilon = J/J(E),
  where J(E) is angular momentum for circular orbit of same energy
  Assumes a spherical potential
  '''

  GMr = const_G * menc / (rsort + sys.float_info.min)

  epsilon = np.zeros(len(do_stars), dtype=vel.dtype)
  rcirc = np.zeros(len(do_stars), dtype=vel.dtype)

  for j in range(len(do_stars)):
    if not do_stars[j]: continue

    r = np.linalg.norm(pos[j])
    v = np.linalg.norm(vel[j])
    J = np.linalg.norm(np.cross(pos[j], vel[j]))
  
    if r == 0.:
      # This is the most bound particle
      epsilon[j] = 1.
      rcirc[j] = r
      continue
  
    # Energy of current orbit
    ibest = np.abs(rsort-r).argmin()
    E = 0.5 * v**2 + (-GMr[ibest] + potential_outer[ibest])
  
    # Bad estimate of energy, perhaps due to spherical potential assumption
    # Just revert to circular orbit
    if E >= 0:
      epsilon[j] = 1.
      rcirc[j] = r
      continue

    # Solve for circular orbit: E = 0.5 GM(r)/r + (-GM(r)/r + V_out)
    dE = E + 0.5 * GMr - potential_outer
    dE_cross = dE[:-1] * dE[1:]

    # Looking for first solution that brackets 0
    isol = np.argmax(dE_cross < 0)

    # Is there a valid solution?
    if dE_cross[isol] < 0:
      # Interpolate to the zero point
      Mc = np.interp(0., dE[isol:isol+2], menc[isol:isol+2])
      Vout = np.interp(0., dE[isol:isol+2], potential_outer[isol:isol+2])

    else:
      # Possibly the baryon particle limiting means we can't get to large enough
      # radii to find the right solution. These don't really matter anyway as
      # DF time will be very long
      isol = len(menc)-1
      Mc = menc[-1]
      Vout = 0.
  
    rc = 0.5 * const_G * Mc / (Vout-E)
    vc = (const_G * Mc / rc)**0.5
    JE = rc * vc

    epsilon[j] = min(J/JE, 1.)
    rcirc[j] = rc

  return epsilon, rcirc

def enclosed_velocity_dispersion(rcl, radii, vel):
  '''
  Velocity dispersion within particle radius
  '''

  rmask = radii < rcl

  if np.sum(rmask) < Min_Nngb:
    # Not enough particles within the radius, increase r to get more
    # (or use everything available)
    rsort = radii[:Min_Nngb]
    vsort = vel[:Min_Nngb]

  else:
    rsort = radii[rmask]
    vsort = vel[rmask]

  Nngb = len(vsort)

  dv = np.sum(vsort, axis=0)
  dv2 = np.sum(vsort**2)

  norm = 1./float(Nngb)
  velStdv = norm * dv2 - norm * norm * np.sum(dv**2)

  if velStdv>0:
    velStdv = np.sqrt(velStdv)
  else:
    print('ERROR: velStdv<0,', velStdv)
    print('Nngb', Nngb)
    print('dv, dv2 = ', dv, dv2)
    sys.stdout.flush()

  # 1D velocity dispersion, sigma/sqrt(3)
  velStdv /= 1.73205

  return velStdv, rsort[-1]

INV_SQRT_PI = 1./np.sqrt(np.pi)
def B(X):
  return erf(X) - (2. * X * INV_SQRT_PI * np.exp(-X**2))

def get_df_timescale(Mgal, Mcl, r, epsilon, velStdv, const_G):
  '''
  Calculate dynamical friction timescale for GC in halo
  Follows calculation in Lacey & Cole 1993, MNRAS, 262, 627L
  '''

  # Velocity dispersion correction
  Vcirc = (const_G * Mgal / r)**0.5
  BX = B(Vcirc / (1.4142 * velStdv))
  alpha = 0.7071 / BX # 2**0.5/2/B(X)

  # Eccentricity correction
  f_e = epsilon**0.78

  Lambda = 1. + Mgal / Mcl

  return alpha * f_e * velStdv * r**2 / (const_G * Mcl * np.log(Lambda))

def get_removed_clusters(args, numthreads):
  '''
  Calculate dynamical friction properties and list of removed clusters

  Parameters
  ----------
  args : argparse Namespace
  numthreads : Number of threads for thread pool

  Returns
  -------
  clusters : Dictionary containing dynamical friction properties for GCs
  '''

  snapnum = args.snapnum
  loadall = not args.partload
  snapPotentials = args.snapPotentials
  eccNbaryon = args.eccNbaryon

  snapfile = f"{args.path}/SOAP/{args.basename}_with_SOAP_membership_{args.snapnum:04}.hdf5"
  snapshot = h5py.File(snapfile, 'r')

  # Set up a dictionary for the cluster properties
  #   removed: Were GCs removed by dynamical friction?
  #   snapNumRemoved: Snapshot at which GCs were removed (i.e. this snapshot)
  #   tfric: Dynamical friction timescale for each GC
  #   ageRemoved: Age at which GCs were removed
  #   massRemoved: Mass at which GCs were removed
  #   TrackId: HBT track ID in which GC resides
  clusters = {}
  clusters['tfric'] = np.array([])
  clusters['removed'] = np.array([], dtype=int)
  clusters['snapNumRemoved'] = np.array([], dtype=int)
  clusters['ageRemoved'] = np.array([])
  clusters['massRemoved'] = np.array([])
  clusters['TrackId'] = np.array([], dtype=int)
  
  itypes = {}
  PartTypeNames = snapshot['Header/PartTypeNames'][:]
  for i in range(len(PartTypeNames)):
    itypes[PartTypeNames[i].decode()] = i

  iDM = itypes['DM']
  istar = itypes['Stars']
  iBH = itypes['BH']

  Ntype = {}
  for ptype in ['Gas', 'DM', 'Stars', 'BH']:
    ipart = itypes[ptype]
    Ntype[ipart] = snapshot['Header'].attrs['NumPart_Total'][ipart]
 
  # If there's no stars yet we can stop here
  if (snapshot['Header'].attrs['NumPart_Total'][istar] == 0):
    print('No stars in snapshot')
    snapshot.close()
    return clusters

  mass_dtype = snapshot['PartType4/Masses'].dtype
  pos_dtype = snapshot['PartType4/Coordinates'].dtype
  vel_dtype = snapshot['PartType4/Velocities'].dtype
  pot_dtype = snapshot['PartType4/Potentials'].dtype
  ids_dtype = snapshot['PartType4/ParticleIDs'].dtype

  timing = {}
  timing['I/O'] = 0.
  timing['Reprocessing'] = 0.
  timing['Centring'] = 0.
  timing['EnclosedMasses'] = 0.
  timing['Eccentricities'] = 0.
  timing['DFCalculation'] = 0.
  timing['TotalHaloes'] = 0.

  start = time.time()

  # Get some parameters/constants
  ascale = snapshot['Header'].attrs['Scale-factor'][0]
  simTime = snapshot['Header'].attrs['Time'][0]
  boxsize = snapshot['Header'].attrs['BoxSize'] * ascale
  const_G = snapshot['PhysicalConstants/InternalUnits'].attrs['newton_G'][0]
  const_Msun = snapshot['PhysicalConstants/InternalUnits'].attrs['solar_mass'][0]

  gc_shape = snapshot[f'PartType{istar}/GCs_Masses'].shape

  # Now properly set up the cluster arrays
  # Initially pick a number much longer than age of universe for tfric
  clusters['tfric'] = 1000. * simTime * np.ones( gc_shape, dtype=np.float32 )
  clusters['removed'] = np.zeros(gc_shape, dtype=np.uint8)
  clusters['snapNumRemoved'] = -1 * np.ones(gc_shape, dtype=np.int16)
  clusters['ageRemoved'] = -1. * np.ones(gc_shape, dtype=np.float32)
  clusters['massRemoved'] = -1. * np.ones(gc_shape, dtype=np.float32)
  clusters['TrackId'] = np.zeros(gc_shape, dtype=np.int32)

  # Maximum inspiral time. Conservatively, twice the age of universe
  Tmax = 2. * simTime

  # Get the subhaloes
  soap_catalogue = f"{args.path}/SOAP/halo_properties_{args.snapnum:04}.hdf5"
  print('Loading haloes from', soap_catalogue)
  sys.stdout.flush()

  subhaloes = {}
  with h5py.File(soap_catalogue, 'r') as f:
    subhaloes['HaloCatalogueIndex'] = np.array(f['InputHalos/HaloCatalogueIndex'])
    subhaloes['TrackId'] = np.array(f['InputHalos/HBTplus/TrackId'])
    subhaloes['Nbound'] = np.array(f['InputHalos/NumberOfBoundParticles'])

    subhaloes['NboundType'] = np.zeros((len(subhaloes['Nbound']), len(PartTypeNames)), dtype=int)
    subhaloes['NboundType'][:, itypes['Gas']] = np.array(f['BoundSubhalo/NumberOfGasParticles'])
    subhaloes['NboundType'][:, itypes['DM']] = np.array(f['BoundSubhalo/NumberOfDarkMatterParticles'])
    subhaloes['NboundType'][:, itypes['Stars']] = np.array(f['BoundSubhalo/NumberOfStarParticles'])
    subhaloes['NboundType'][:, itypes['BH']] = np.array(f['BoundSubhalo/NumberOfBlackHoleParticles'])

    subhaloes['CentreOfPotential'] = np.array(f['InputHalos/HaloCentre']) * ascale

    # Velocities stored in physical units 
    subhaloes['CentreOfMassVelocity'] = np.array(f['BoundSubhalo/CentreOfMassVelocity'])

  # Nothing to do if there's no bound subhaloes or subhaloes with stars
  if len(subhaloes['HaloCatalogueIndex']) > 0:
    Nhaloes = np.sum(subhaloes['NboundType'][:,istar] > 0)
  else:
    Nhaloes = 0

  print(f'{Nhaloes} in SOAP with stars')
  sys.stdout.flush()
  if Nhaloes == 0:
    snapshot.close()
    return clusters

  ##### Load snapshot particle data #####

  stars_in_haloes = snapshot[f'PartType{istar}/HaloCatalogueIndex'][:] >= 0
  has_gcs = snapshot[f'PartType{istar}/GCs_NumberOfClusters'][:] > 0

  if not stars_in_haloes.any() or not has_gcs.any():
    snapshot.close()
    return clusters

  print('Loading particle halo IDs')
  sys.stdout.flush()

  # Full lists
  All_HaloIndex = {}
  for ptype in ['Gas', 'DM', 'Stars', 'BH']:
    ipart = itypes[ptype]
    All_HaloIndex[ptype] = snapshot[f'PartType{ipart}/HaloCatalogueIndex'][:]

  if loadall:
    # Load all particles at once

    #Bound_IDs = {}
    Bound_masses = {}
    Bound_pos = {}
    Bound_vel = {}
    Bound_pots = {}
    Bound_HaloIndex = {}

    for ptype in ['Gas', 'DM', 'Stars', 'BH']:
      ipart = itypes[ptype]
      if Ntype[ipart] == 0: continue
 
      print('Loading', ptype)
      sys.stdout.flush()
 
      if ptype == 'BH':
        MassType = 'DynamicalMasses'
      else:
        MassType = 'Masses'
 
      group = snapshot[f'PartType{ipart}']

      # Limit to only particles in haloes
      in_halo = All_HaloIndex[ptype] >= 0
      Bound_HaloIndex[ptype] = All_HaloIndex[ptype][in_halo]
 
      #Bound_IDs[ptype] = group['ParticleIDs'][in_halo]
      Bound_masses[ptype] = \
          load_physical_data(group, MassType, ascale)[in_halo]
      Bound_pos[ptype] = \
          load_physical_data(group, 'Coordinates', ascale)[in_halo]

      if snapPotentials:
        if 'Potentials' in group and ptype in ['DM', 'Stars', 'BH']:
          Bound_pots[ptype] = \
              load_physical_data(group, 'Potentials', ascale)[in_halo]
 
      if ptype in ['DM', 'Stars', 'BH']:
        Bound_vel[ptype] = \
            load_physical_data(group, 'Velocities', ascale)[in_halo]

      del in_halo
 
    group = snapshot[f'PartType{istar}']
    Bound_GCs_Masses = load_physical_data(group, 'GCs_Masses', ascale)[stars_in_haloes]

    Stars_HaloIndex = Bound_HaloIndex['Stars']
    #Stars_IDs = Bound_IDs['Stars']
    #Stars_Pos = Bound_pos['Stars']
    #Stars_Vel = Bound_vel['Stars']

    del All_HaloIndex

  else:
    # Just load IDs, indexed loading of subhalo parts to save memory, at the
    # expense of increased I/O time

    #Bound_IDs = {}
    Bound_HaloIndex = {}
    for ptype in ['Gas', 'DM', 'Stars', 'BH']:
      ipart = itypes[ptype]
      if Ntype[ipart] == 0: continue

      # Limit to only particles in haloes
      in_halo = All_HaloIndex[ptype] >= 0
      Bound_HaloIndex[ptype] = All_HaloIndex[ptype][in_halo]
      #Bound_IDs[ipart] = snapshot[f'PartType{ipart}/ParticleIDs'][in_halo]

      del in_halo

    Stars_HaloIndex = Bound_HaloIndex['Stars']

  snapshot.close()

  # Location of stars in full snapshot array
  Stars_SnapIndex = np.arange(len(stars_in_haloes))[stars_in_haloes]

  has_gcs = has_gcs[stars_in_haloes]

  del stars_in_haloes

  # Number of GCs in each subhalo
  unique_haloes = np.unique(Stars_HaloIndex)
  subhaloes['Ngcs'] = np.zeros(len(subhaloes['Nbound']), dtype=int)
  for haloIdx in unique_haloes:

    # Only subhaloes with GCs
    Ngcs = np.sum(has_gcs[Bound_HaloIndex['Stars'] == haloIdx])
    if Ngcs == 0: continue

    subhaloes['Ngcs'][ subhaloes['HaloCatalogueIndex'] == haloIdx ] = Ngcs

  del has_gcs

  Nhaloes = np.sum(subhaloes['Ngcs'] > 0)
  print(f"{Nhaloes} subhaloes with GCs")

  timing['I/O'] += (time.time() - start) / 60.
  print(f"Initial load took {timing['I/O']} mins")

  current, peak = tracemalloc.get_traced_memory()
  print(f'Current memory usage: {current/1024.**3:.5g} GB')
  sys.stdout.flush()


  timing_haloes = np.zeros(len(subhaloes['Nbound']), dtype=np.float32)

  def df_one_subhalo(isub):
    '''
    Set up dynamical friction calculation for one halo to parallelise over
    '''

    # Timing
    start_halo = time.time()
    start = time.time()
    thalo_reproc = 0.
    thalo_io = 0.
    thalo_cent = 0.
    thalo_menc = 0.
    thalo_ecc = 0.
    thalo_df = 0.

    HaloIdx = subhaloes['HaloCatalogueIndex'][isub]
    Nbound = subhaloes['Nbound'][isub]
    Ntype = subhaloes['NboundType'][isub]
    Nstar = Ntype[istar]

    # Only resolved haloes with stars
    if (Nbound == 0) or (Nstar == 0):
      return

    # Allocate space for subhalo particle properties
    SH_type = np.zeros(Nbound, dtype=np.int8)
    SH_masses = np.zeros(Nbound, dtype=mass_dtype)
    SH_pos = np.zeros((Nbound, 3), dtype=pos_dtype)
    SH_vel = np.zeros((Ntype[iDM]+Ntype[istar]+Ntype[iBH], 3), dtype=vel_dtype)
    #SH_IDs = np.zeros(Nbound, dtype=ids_dtype)
    #SH_Star_IDs = np.array([], dtype=ids_dtype)
    if snapPotentials:
      SH_pots = np.ones(Ntype[iDM]+Ntype[istar]+Ntype[iBH], dtype=pot) * np.inf

    thalo_reproc += time.time() - start

    # Match the subhalo particles and get a list of all particle properties
    Ngcs = 0
    if loadall:
      # Data already loaded so just match particles
      start = time.time()

      mask = Bound_HaloIndex['Stars'] == HaloIdx
      GCs_Masses = Bound_GCs_Masses[mask]

      # Does this subhalo have any clusters?
      Ngcs = np.sum(GCs_Masses > 0)
      if Ngcs > 0:

        # Get the subhalo particles
        upto = 0
        upto_vel = 0
        for ptype in ['Gas', 'DM', 'Stars', 'BH']:
          ipart = itypes[ptype]
          if Ntype[ipart] == 0: continue

          mask = Bound_HaloIndex[ptype] == HaloIdx
          SH_type[upto:upto+Ntype[ipart]] = ipart * np.ones(Ntype[ipart], dtype=np.int8)
          SH_masses[upto:upto+Ntype[ipart]] = Bound_masses[ptype][mask]
          SH_pos[upto:upto+Ntype[ipart]] = Bound_pos[ptype][mask]
          if snapPotentials:
            if ptype in ['DM', 'Stars', 'BH']:
              SH_pots[upto_vel:upto_vel+Ntype[ipart]] = Bound_pots[ptype][mask]

          if ptype in ['DM', 'Stars', 'BH']:
            SH_vel[upto_vel:upto_vel+Ntype[ipart]] = \
                Bound_vel[ptype][mask]
            upto_vel += Ntype[ipart]

          #if ptype == 'Stars':
          #  SH_Star_IDs = Stars_IDs[ Stars_HaloIndex == HaloIdx ]

          upto += Ntype[ipart]

      thalo_reproc += time.time() - start

    else:
      # Load particles for this halo from file

      start = time.time()

      snapshot = h5py.File(snapfile, 'r')

      mask = All_HaloIndex['Stars'] == HaloIdx
      p_ind = np.arange(len(mask))[mask]

      group = snapshot[f'PartType{istar}']
      GCs_Masses = load_physical_data(group, 'GCs_Masses', ascale, p_ind)

      # Does this subhalo have any clusters?
      Ngcs = np.sum(GCs_Masses > 0)
      if Ngcs > 0:

        upto = 0
        upto_vel = 0
        for ptype in ['Gas', 'DM', 'Stars', 'BH']:

          ipart = itypes[ptype]
          if Ntype[ipart] == 0: continue

          group = snapshot[f'PartType{ipart}']

          ## Match the snapshot and subhalo particles
          mask = All_HaloIndex[ptype] == HaloIdx
          p_ind = np.arange(len(mask))[mask]

          if ptype == 'BH':
            MassType = 'DynamicalMasses'
          else:
            MassType = 'Masses'
 
          SH_type[upto:upto+Ntype[ipart]] = ipart * np.ones(Ntype[ipart], dtype=np.int8)
          #SH_IDs[upto:upto+Ntype[ipart]] = group['ParticleIDs'][p_ind]
          SH_masses[upto:upto+Ntype[ipart]] = \
              load_physical_data(group, MassType, ascale, p_ind)
          SH_pos[upto:upto+Ntype[ipart]] = \
              load_physical_data(group, 'Coordinates', ascale, p_ind)
          if snapPotentials:
            if ptype in ['DM', 'Stars', 'BH']:
              SH_pots[upto_vel:upto_vel+Ntype[ipart]] = \
                  load_physical_data(group, 'Potentials', ascale, p_ind)
 
          if ptype in ['DM', 'Stars', 'BH']:
            SH_vel[upto_vel:upto_vel+Ntype[ipart]] = \
                load_physical_data(group, 'Velocities', ascale, p_ind)
            upto_vel += Ntype[ipart]

          #if ptype == 'Stars':
          #  #SH_Star_IDs = np.array(All_IDs[ipart][p_ind], copy=True)
          #  SH_Star_IDs = SH_IDs[upto:upto+Ntype[ipart]]

          upto += Ntype[ipart]
 
          del p_ind

      snapshot.close()

      thalo_io += time.time() - start

    # Does this subhalo have any clusters?
    if Ngcs == 0:
      end_halo = time.time()
      with timing_lock:
        timing['TotalHaloes'] += (end_halo - start_halo) / 60.
        timing['Reprocessing'] += thalo_reproc / 60.
        timing['I/O'] += thalo_io / 60.
        timing['Centring'] += thalo_cent / 60.
        timing['EnclosedMasses'] += thalo_menc / 60.
        timing['Eccentricities'] += thalo_ecc / 60.
        timing['DFCalculation'] += thalo_df / 60.
        timing_haloes[isub] = (end_halo - start_halo) / 60.
      return
 
    start = time.time()

    # Location of subhalo stars in full snapshot array
    SH_star_snapIdx = Stars_SnapIndex[Stars_HaloIndex == HaloIdx]

    clusters['TrackId'][SH_star_snapIdx] = subhaloes['TrackId'][isub]
 
    if snapPotentials:
      # Use particle snapshot potentials for centre of potential
      cofp = SH_pos[ SH_pots.argmin() ]
      cofv = SH_vel[ SH_pots.argmin() ]

      print("COP from particle potentials: ",cofp)
      print("COP velocity from particle potentials: ",cofv)

      print("COP from SOAP: ", subhaloes['CentreOfPotential'][isub])
      print("COM velocity from SOAP: ", subhaloes['CentreOfMassVelocity'][isub])

      del SH_pots
 
    else:
      # Use subhalo catalogue for centres
      cofp = subhaloes['CentreOfPotential'][isub]
      cofv = subhaloes['CentreOfMassVelocity'][isub]

    # centre positions and velocities on subhalo centre of potential
    SH_pos = centre_periodic_positions(SH_pos, cofp, boxsize)
    SH_vel -= cofv
    SH_radii = np.linalg.norm(SH_pos, axis=1)

    # Radii of the stars
    SH_star_radii = SH_radii[ SH_type == istar ]

    thalo_cent += time.time() - start
    start = time.time()
 
    # Prepare for the DF timescale calculations
 
    # Sort subhalo particles by radius
    # We assume a spherically symmetric potential, which is probably ok for
    # a DM-dominated system
 
    # Particles with velocities first
    vel_mask = np.bitwise_or(SH_type == iDM, SH_type == istar, SH_type == iBH)
    SH_radii_vel = SH_radii[vel_mask]
    sortIdx = SH_radii_vel.argsort()
    SH_radii_vel = SH_radii_vel[sortIdx]
    SH_type_vel = SH_type[vel_mask][sortIdx]
    SH_vel = SH_vel[sortIdx]
 
    # Now all the particles
    sortIdx = SH_radii.argsort()
    SH_radii = SH_radii[sortIdx]
    SH_masses = SH_masses[sortIdx]
    SH_type = SH_type[sortIdx]
    SH_pos = SH_pos[sortIdx]
    #SH_IDs = SH_IDs[sortIdx]

    sortIdx = SH_star_radii.argsort()
    GCs_Masses = GCs_Masses[sortIdx]
    SH_star_snapIdx = SH_star_snapIdx[sortIdx]
    SH_star_radii = SH_star_radii[sortIdx]
    #SH_Star_IDs = SH_Star_IDs[sortIdx]

    thalo_reproc += time.time() - start
    start = time.time()
 
    # Enclosed mass
    menc = np.zeros(len(SH_masses), dtype=SH_masses.dtype)
    menc[0] = 0.
    for i in range(1, len(menc)):
      menc[i] = menc[i-1] + SH_masses[i]
 
    # Approx. upper limit to the distance a cluster can inspiral from
    # Used to limit to calculations worth doing
    Mtest = 1.1 * np.max(GCs_Masses)
    Rmax = SH_radii[-1]
    n = len(menc)
    dn = max(1, int(n/100)) # check up to ~100 radii
    for i in range(dn-1, len(menc), dn):
      if (menc[i] == 0.) or (SH_radii[i] == 0.):
        Tdf = 0.
      else:
        Vcirc = (const_G * menc[i] / SH_radii[i])**0.5
        Lambda = 1.+menc[i]/Mtest
        Tdf = 1.17 * Vcirc * SH_radii[i]**2 / (const_G * Mtest * np.log(Lambda))

      if Tdf > Tmax:
        Rmax = SH_radii[i]
        break

    thalo_menc += time.time() - start
    start = time.time()
 
    # Get a list of stars to process
    do_stars = np.zeros(Nstar, dtype=bool)
    for i in range(Nstar):
 
      # Too far, don't bother
      if SH_star_radii[i] > Rmax: continue
 
      # Does this particle have any clusters?
      if not (GCs_Masses[i] > 0).any(): continue
 
      if SH_star_radii[i] == 0:
        # Can stop here. t_df = 0 for r = 0
        for j in range( gc_shape[1] ):
          if GCs_Masses[i,j] > 0:
            clusters['tfric'][SH_star_snapIdx[i], j] = 0.

      else:
        do_stars[i] = True

    print(f"{isub}/{Nhaloes}: Rmax={Rmax:.4g}, Mmax={Mtest/1.1:.4g}, Nstars={Nstar}, do_stars={np.sum(do_stars)}, too_far={np.sum(SH_star_radii > Rmax)}")
    sys.stdout.flush()
 
    thalo_reproc += time.time() - start
    start = time.time()
 
    if np.sum(do_stars) == 0:
      end_halo = time.time()
      with timing_lock:
        timing['TotalHaloes'] += (end_halo - start_halo) / 60.
        timing['Reprocessing'] += thalo_reproc / 60.
        timing['I/O'] += thalo_io / 60.
        timing['Centring'] += thalo_cent / 60.
        timing['EnclosedMasses'] += thalo_menc / 60.
        timing['Eccentricities'] += thalo_ecc / 60.
        timing['DFCalculation'] += thalo_df / 60.
        timing_haloes[isub] = (end_halo - start_halo) / 60.
      return
 
    # Outer component of potential, assuming spherical symmetry
    # -G int_r^inf dM(r')/r'
    potential_outer = np.zeros(len(SH_masses), dtype=SH_masses.dtype)
    for i in range(len(SH_masses)-2,-1,-1):
      potential_outer[i] = \
          potential_outer[i+1] - const_G * SH_masses[i+1] / SH_radii[i+1]
 
    # get eccentricities for all stars at once
    dx = centre_periodic_positions(
        SH_pos[ SH_type == istar ], cofp, boxsize)
    dv = SH_vel[ SH_type_vel == istar ] - cofv
 
    # If there are enough baryons to sample the potential, just use them
    if (Ntype[istar]+Ntype[itypes['Gas']]) > eccNbaryon:
      # Use only stars if there are enough
      if Ntype[istar] > eccNbaryon:
        mask = (SH_type == istar)
      else:
        mask = np.bitwise_or(SH_type == itypes['Gas'], SH_type == istar)

      # Make sure the very central particles are there as well
      mask[:100] = True

      epsilon, rcirc = eccentricity(do_stars, dx, dv, const_G, SH_radii[mask],
          menc[mask], potential_outer[mask])

    else:
      epsilon, rcirc = eccentricity(
          do_stars, dx, dv, const_G, SH_radii, menc, potential_outer)

    thalo_ecc += time.time() - start
    start = time.time()
 
    # Now loop over the stars in subhalo to calculate DF times for clusters
    for i in range(Nstar):
 
      if not do_stars[i]: continue
 
      # Enclosed mass of galaxy
      Mgal = np.sum(SH_masses[SH_radii <= rcirc[i]])
 
      # Enclosed velocity dispersion
      # Use combination of DM+stars for vel disp. If there's enough, just use
      # stars for speed
      tmask = (SH_radii_vel < rcirc[i]) & (SH_type_vel == istar)
      if np.sum(tmask) >= Min_Nngb:
        # Enough stars, just use them
        velStdv, vel_rmax = enclosed_velocity_dispersion(
            rcirc[i], SH_radii_vel[tmask], SH_vel[tmask])
      else:
        # Not enough stars, so use combination of DM+stars
        velStdv, vel_rmax = enclosed_velocity_dispersion(
            rcirc[i], SH_radii_vel, SH_vel)
 
      # Do all clusters of this particle
      part_GC_masses = GCs_Masses[i]
      for j in range( gc_shape[1] ):
        if part_GC_masses[j] <= 0: continue

        if Mgal == 0:
          clusters['tfric'][SH_star_snapIdx[i], j] = 0.

        else:
          clusters['tfric'][SH_star_snapIdx[i], j] = get_df_timescale(
              Mgal, part_GC_masses[j], rcirc[i], epsilon[i], velStdv, const_G)

    thalo_df += time.time() - start
    end_halo = time.time()

    with timing_lock:
      timing['TotalHaloes'] += (end_halo - start_halo) / 60.
      timing['Reprocessing'] += thalo_reproc / 60.
      timing['I/O'] += thalo_io / 60.
      timing['Centring'] += thalo_cent / 60.
      timing['EnclosedMasses'] += thalo_menc / 60.
      timing['Eccentricities'] += thalo_ecc / 60.
      timing['DFCalculation'] += thalo_df / 60.
      timing_haloes[isub] = (end_halo - start_halo) / 60.

    return

  # Loop over all the subhaloes with GCs
  submask = subhaloes['Ngcs'] > 0
  do_halo = np.arange(len(subhaloes['Ngcs']))[submask]
  #nsort = subhaloes['Nbound'][submask].argsort()[::-1] # sorting to do largest haloes first
  #do_halo = do_halo[nsort]

  print('Running DF...')
  sys.stdout.flush()

  start_pardf = time.time()

  with ThreadPoolExecutor(max_workers=numthreads) as exe:
    # Do the hard work
    results = exe.map(df_one_subhalo, do_halo)

    for result in results:
      if not result is None:
        print(result)

  end_pardf = time.time()

  snapshot = h5py.File(snapfile, 'r')
  group = snapshot[f'PartType{istar}']
  if 'Ages' in group:
    Stars_Ages = load_physical_data(group, 'Ages', ascale)

  else:
    # SOAP doesn't use snipshots

    ## If snipshot doesn't have ages, use a future snapshot

    ## Check if using subdirs
    #hasSubdir = False
    #if (snapshot.filename ==
    #    f'{args.path}/{args.basename}_{snapnum:04}/' +
    #    f'{args.basename}_{snapnum:04}.hdf5'):
    #  hasSubdir = True

    ## Get the last snapshot index
    #lastsnap = -1 # dummy starter
    #if hasSubdir:
    #  # Check path for subdirectories
    #  for snapdir in os.listdir(args.path):

    #    # Match the filename base, and make sure is directory
    #    if (fnmatch.fnmatch(snapdir, f'{args.basename}_*') and
    #        os.path.isdir(f'{args.path}/{snapdir}')):

    #      snap = int(snapdir.split('_')[1])
    #      if snap > lastsnap:
    #        lastsnap = snap

    #else:
    #  # Check path for hdf5 files
    #  for snapfile in os.listdir(args.path):

    #    # Match the filename, and make sure is a file
    #    if (fnmatch.fnmatch(snapfile, f'{args.basename}_*.hdf5') and
    #        os.path.isfile(f'{args.path}/{snapfile}')):

    #      snap = int(snapfile.split('.hdf5')[0].split('_')[1])
    #      if snap > lastsnap:
    #        lastsnap = snap

    #foundSnap = False
    #for isnap in range(snapnum+1, lastsnap+1):

    #  if hasSubdir:
    #    snapfile = f'{args.path}/{args.basename}_{isnap:04}/' + \
    #               f'{args.basename}_{isnap:04}.hdf5'
    #  else:
    #    snapfile = f'{args.path}/{args.basename}_{isnap:04}.hdf5'

    #  nextsnap = h5py.File(snapfile, 'r')

    #  if nextsnap['Header'].attrs['SelectOutput'].decode() == 'Snipshot':
    #    # Looking for snapshot, not snipshot
    #    nextsnap.close()
    #    continue

    #  print(f'Getting stellar ages from snapshot {isnap}')
    #  foundSnap = True
    #  break

    #if foundSnap:
    #  # Ok, found a snapshot. Now get the ages
    #  Future_IDs = nextsnap[f'PartType{istar}/ParticleIDs'][:]
    #  Future_Ages = load_physical_data(
    #      nextsnap[f'PartType{istar}'], 'Ages', ascale)

    #  # Correct future ages to current time
    #  futureTime = nextsnap['Header'].attrs['Time'][0]
    #  Future_Ages -= (futureTime - simTime)

    #  _, snip_ind, snap_ind = np.intersect1d(Stars_IDs, Future_IDs,
    #      assume_unique=True, return_indices=True)

    #  Stars_Ages = np.zeros(len(Stars_IDs), dtype=Future_Ages.dtype)
    #  Stars_Ages[snip_ind] = Future_Ages[snap_ind]

    #  # Also need to store Age attrs for later
    #  clusters['Age_attrs'] = {}
    #  for key in nextsnap['PartType4/Ages'].attrs.keys():
    #    clusters['Age_attrs'][key] = nextsnap['PartType4/Ages'].attrs[key]

    #  nextsnap.close()

    #else:
    #  print("Couldn't find a future snapshot for stellar ages, exiting...")
    #  exit(1)

    print("Couldn't find a future snapshot for stellar ages, exiting...")
    exit(1)

  # Reload the full GC masses array
  group = snapshot[f'PartType{istar}']
  All_GCs_Masses = load_physical_data(group, 'GCs_Masses', ascale)

  snapshot.close()

  # If dynamical friction timescale is shorter than cluster age then cluster 
  # is removed/considered disrupted
  # Assumes particle has been in the same halo the whole time. DF time will
  # generally significantly increase if the host galaxy is accreted, so should
  # mainly only affect the in situ GCs
  GC_ages = (np.ones(gc_shape, dtype=Stars_Ages.dtype).T * np.array(Stars_Ages)).T

  df_mask = (clusters['tfric'] < GC_ages)
  clusters['removed'][df_mask] = 1
  clusters['snapNumRemoved'][df_mask] = snapnum
  clusters['massRemoved'][df_mask] = np.array(All_GCs_Masses)[df_mask]
  clusters['ageRemoved'][df_mask] = GC_ages[df_mask]

  mask = All_GCs_Masses > 0
  print('%d / %d clusters removed' %
      (np.sum(clusters['removed'][mask]), np.sum(mask)))

  print('Timing:')
  for key in timing.keys():
    if key == 'TotalHaloes': continue
    print('  %s : %.4g mins' % (key, timing[key]))
  print('Parallel CPU %g mins, wallclock %g mins' %
       (timing['TotalHaloes'], (end_pardf-start_pardf)/60.))

  timing_haloes[::-1].sort()
  print('Top 5 haloes:', timing_haloes[:5], 'mins')
  sys.stdout.flush()
      
  return clusters

def write_clusters(clusters, args):
  '''
  Write dynamical friction data to HDF5 file

  Parameters
  ----------
  clusters : The clusters dictionary
  args : argparse Namespace
  '''

  output = args.output
  snapnum = args.snapnum

  snapfile = f"{args.path}/SOAP/{args.basename}_with_SOAP_membership_{args.snapnum:04}.hdf5"
  snapshot = h5py.File(snapfile, 'r')

  # Open for writing
  f_df = h5py.File(f'{output}/df_timescale_{snapnum:04}.hdf5', 'w')

  # Copy headers
  snapshot.copy('Cosmology', f_df)
  snapshot.copy('Header', f_df)
  snapshot.copy('InternalCodeUnits', f_df)
  snapshot.copy('PhysicalConstants', f_df)
  snapshot.copy('Units', f_df)

  if len(clusters['removed']) == 0:
    f_df.close()
    snapshot.close()
    return

  # Age attributes need special handling, as they may not exist in snipshots
  # If we didn't get them from a future snapshot (for snips), get from snapshot
  if not 'Age_attrs' in clusters:
    clusters['Age_attrs'] = {}
    for key in snapshot['PartType4/Ages'].attrs.keys():
      clusters['Age_attrs'][key] = snapshot['PartType4/Ages'].attrs[key]

  # now write data to file
  group = f_df.create_group('PartType4')

  ds = group.create_dataset('ParticleIDs',
      data=snapshot['PartType4/ParticleIDs'][:], dtype='u8', compression="gzip")
  for key in snapshot['PartType4/ParticleIDs'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/ParticleIDs'].attrs[key]

  ds = group.create_dataset('GCs_DF_Removed', data=clusters['removed'],
      dtype='u1', compression="gzip")
  for key in snapshot['PartType4/ParticleIDs'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/ParticleIDs'].attrs[key]
  ds.attrs['Description'] = "Was GC removed by dynamical friction?"

  ds = group.create_dataset('GCs_DF_SnapnumRemoved',
      data=clusters['snapNumRemoved'], dtype='i2', compression="gzip")
  for key in snapshot['PartType4/ParticleIDs'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/ParticleIDs'].attrs[key]
  ds.attrs['Description'] = \
      "Snapshot at which GC was removed by dynamical friction"

  ds = group.create_dataset('GCs_DF_Timescale', data=clusters['tfric'],
      dtype='f4', compression="gzip")
  for key in clusters['Age_attrs'].keys():
    ds.attrs[key] = clusters['Age_attrs'][key]
  ds.attrs['Value stored as physical'] = np.array([1])
  ds.attrs['Description'] = "Dynamical friction timescale"

  ds = group.create_dataset('GCs_DF_AgeRemoved', data=clusters['ageRemoved'],
      dtype='f4', compression="gzip")
  for key in clusters['Age_attrs'].keys():
    ds.attrs[key] = clusters['Age_attrs'][key]
  ds.attrs['Value stored as physical'] = np.array([1])
  ds.attrs['Description'] = "Age at time of dynamical friction removal"

  ds = group.create_dataset('GCs_DF_MassRemoved', data=clusters['massRemoved'],
      dtype='f4', compression="gzip")
  for key in snapshot['PartType4/GCs_Masses'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/GCs_Masses'].attrs[key]
  ds.attrs['Value stored as physical'] = np.array([1])
  ds.attrs['Description'] = "Mass at time of dynamical friction removal"

  ds = group.create_dataset('GCs_DF_TrackId', data=clusters['TrackId'],
      dtype='i4', compression="gzip")
  for key in snapshot['PartType4/ParticleIDs'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/ParticleIDs'].attrs[key]
  ds.attrs['Description'] = "HBT+ TrackId of halo in which GC resides"

  f_df.close()
  snapshot.close()
  return


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--numthreads', action='store', type=int,
    dest='numthreads', default=-1,
    help="Number of threads (default=-1, use _max_workers)")
  parser.add_argument('--path', action='store', type=str,
    dest='path', default='.',
    help="Path to snapshot data (default '.')")
  parser.add_argument('--basename', action='store', type=str,
    dest='basename', default='colibre',
    help="Snapshot basename (default 'colibre')")
  parser.add_argument('--subpath', action='store', type=str,
    dest='subpath', default='.',
    help="Path to subhalo data (default '.')")
  parser.add_argument('--output', action='store', type=str,
    dest='output', default='dynamical_friction',
    help="Output path (default 'dynamical_friction')")
  parser.add_argument('--snapnum', action='store', type=int,
    dest='snapnum', default=0,
    help="Snapshot to use (default=0)")
  parser.add_argument('--skipsnips', action='store_true',
    dest='skipSnips', default=False,
    help="Skip snipshot files?")
  parser.add_argument('--partload', action='store_true',
    dest='partload', default=False,
    help="Try and load all data at once to speed up I/O. Otherwise load "
         "halo particles individually (partial load)")
  parser.add_argument('--snappotentials', action='store_true',
    dest='snapPotentials', default=False,
    help="Use particle snapshot potentials for centre of potential?")
  parser.add_argument('--ecc_nbaryon', action='store', type=int,
    dest='eccNbaryon', default=10000,
    help="Particle number to use only baryons for eccentricities "
         "(default=10000)")
  args = parser.parse_args()

  print('*** Dynamical friction timescales ***')
  print()
  print('pid:', os.getpid())
  print('Snapshot path:', args.path)
  print('basename:', args.basename)
  print('Subhalo path:', args.subpath)
  print('snapnum:', args.snapnum)
  print('Output path:', args.output)
  print('Part load:', args.partload)
  print('Snap potentials:', args.snapPotentials)
  sys.stdout.flush()

  # Track memory usage
  tracemalloc.start()

  # Make the output directory if it doesn't exist yet
  os.system(f'mkdir -pv {args.output}')

  snapfile = f"{args.path}/SOAP/{args.basename}_with_SOAP_membership_{args.snapnum:04}.hdf5"

  print('Loading snapshot from', snapfile)
  sys.stdout.flush()
  with h5py.File(snapfile, 'r') as snapshot:
    if args.skipSnips and \
        (snapshot['Header'].attrs['SelectOutput'].decode() == 'Snipshot'):
      print('Snipshot, exiting...')
      exit()

  # Set maximum thread number
  numthreads = args.numthreads
  if numthreads < 1:
    # Default to system maximum
    exe = ThreadPoolExecutor()
    numthreads = exe._max_workers

  print(f'Using {numthreads} threads')

  # Start the dynamical friction calculation and return list of removed GCs
  start = time.time()
  clusters = get_removed_clusters(args, numthreads)
  print('Dynamical friction took %.5g mins' % ((time.time() - start)/60.))

  print(f'Writing output to {args.output}/df_timescale_{args.snapnum:04}.hdf5')
  sys.stdout.flush()
  write_clusters(clusters, args)

  current, peak = tracemalloc.get_traced_memory()
  print(f'Peak memory usage: {peak/1024.**3:.5g} GB')
  tracemalloc.stop()

