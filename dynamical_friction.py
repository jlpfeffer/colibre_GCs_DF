import sys, os, time
import numpy as np
import h5py
import argparse
import fnmatch
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import tracemalloc
from HBTReader import HBTReader
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

  epsilon = np.zeros(len(do_stars))
  rcirc = np.zeros(len(do_stars))

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

def get_removed_clusters(snapshot, args, numthreads):
  '''
  Calculate dynamical friction properties and list of removed clusters

  Parameters
  ----------
  snapshot : HDF5 snapshot file
  args : argparse Namespace
  numthreads : Number of threads for thread pool

  Returns
  -------
  clusters : Dictionary containing dynamical friction properties for GCs
  '''

  SnapshotPath = args.path
  SnapshotFileBase = args.basename
  SubhaloPath = args.subpath
  snapnum = args.snapnum
  loadall = args.loadall
  snapPotentials = args.snapPotentials
  eccNbaryon = args.eccNbaryon

  # Set up a dictionary for the cluster properties
  #   removed: Were GCs removed by dynamical friction?
  #   snapNumRemoved: Snapshot at which GCs were removed (i.e. this snapshot)
  #   tfric: Dynamical friction timescale for each GC
  #   ageRemoved: Age at which GCs were removed
  #   massRemoved: Mass at which GCs were removed
  #   dfTrackId: HBT track ID in which GC resides
  clusters = {}
  clusters['tfric'] = np.array([])
  clusters['removed'] = np.array([], dtype=int)
  clusters['snapNumRemoved'] = np.array([], dtype=int)
  clusters['ageRemoved'] = np.array([])
  clusters['massRemoved'] = np.array([])
  clusters['dfTrackId'] = np.array([], dtype=int)
  
  itypes = {}
  PartTypeNames = snapshot['Header/PartTypeNames'][:]
  for i in range(len(PartTypeNames)):
    itypes[PartTypeNames[i].decode()] = i

  iDM = itypes['DM']
  istar = itypes['Stars']

  # If there's no stars yet we can stop here
  if (snapshot['Header'].attrs['NumPart_Total'][istar] == 0):
    print('No stars in snapshot')
    return clusters

  timing = {}
  timing['I/O'] = 0.
  timing['Reprocessing'] = 0.
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
  clusters['dfTrackId'] = np.zeros(gc_shape, dtype=np.int32)

  # Maximum inspiral time. Conservatively, twice the age of universe
  Tmax = 2. * simTime

  # Get the subhaloes
  print('Loading subhaloes from', SubhaloPath)
  sys.stdout.flush()
  hbt = HBTReader(SubhaloPath)
  subhaloes = hbt.LoadSubhalos(snapnum)

  # Nothing to do if there's no bound subhaloes or subhaloes with stars
  if len(subhaloes) > 0:
    Nhaloes = np.sum(subhaloes['NboundType'][:,istar] > 0)
  else:
    Nhaloes = len(subhaloes)
  print(f'{Nhaloes} haloes to process')
  sys.stdout.flush()
  if Nhaloes == 0:
    return clusters

  # Load subhalo particles
  subparts = hbt.LoadParticles(snapnum)

  # Load snapshot particle data

  if loadall:
    # Load all particles at once

    All_IDs = {}
    All_masses = {}
    All_pos = {}
    All_vel = {}
    All_pots = {}
 
    for ptype in ['Gas', 'DM', 'Stars', 'BH']:
      ipart = itypes[ptype]
      Ntype = snapshot['Header'].attrs['NumPart_Total'][ipart]
      if Ntype == 0: continue
 
      print('Loading', ptype)
      sys.stdout.flush()
 
      if ptype == 'BH':
        MassType = 'DynamicalMasses'
      else:
        MassType = 'Masses'
 
      group = snapshot[f'PartType{ipart}']
 
      All_IDs[ptype] = group['ParticleIDs'][:]
      All_masses[ptype] = \
          load_physical_data(group, MassType, ascale)
      All_pos[ptype] = \
          load_physical_data(group, 'Coordinates', ascale)

      if snapPotentials:
        All_pots[ptype] = \
            load_physical_data(group, 'Potentials', ascale)
 
      if ptype in ['DM', 'Stars']:
        All_vel[ptype] = \
            load_physical_data(group, 'Velocities', ascale)
 
    Stars_IDs = np.array(All_IDs['Stars'], copy=True)
    Stars_Pos = np.array(All_pos['Stars'], copy=True)
    Stars_Vel = np.array(All_vel['Stars'], copy=True)

  else:
    # Just load IDs, indexed loading of subhalo parts to save memory, at the
    # expense of increased I/O time

    All_IDs = {}
    for ptype in ['Gas', 'DM', 'Stars', 'BH']:
      ipart = itypes[ptype]
      All_IDs[ipart] = snapshot[f'PartType{ipart}/ParticleIDs'][:]

    group = snapshot[f'PartType{istar}']
    Stars_IDs = group['ParticleIDs'][:]
    Stars_Pos = load_physical_data(group, 'Coordinates', ascale)
    Stars_Vel = load_physical_data(group, 'Velocities', ascale)

  group = snapshot[f'PartType{istar}']
  GCs_Masses = load_physical_data(group, 'GCs_Masses', ascale)

  if 'Ages' in group:
    Stars_Ages = load_physical_data(group, 'Ages', ascale)

  else:
    # If snipshot doesn't have ages, use a future snapshot

    # Check if using subdirs
    hasSubdir = False
    if (snapshot.filename ==
        f'{SnapshotPath}/{SnapshotFileBase}_{snapnum:04}/' +
        f'{SnapshotFileBase}_{snapnum:04}.hdf5'):
      hasSubdir = True

    # Get the last snapshot index
    lastsnap = -1 # dummy starter
    if hasSubdir:
      # Check path for subdirectories
      for snapdir in os.listdir(SnapshotPath):

        # Match the filename base, and make sure is directory
        if (fnmatch.fnmatch(snapdir, f'{SnapshotFileBase}_*') and
            os.path.isdir(f'{SnapshotPath}/{snapdir}')):

          snap = int(snapdir.split('_')[1])
          if snap > lastsnap:
            lastsnap = snap

    else:
      # Check path for hdf5 files
      for snapfile in os.listdir(SnapshotPath):

        # Match the filename, and make sure is a file
        if (fnmatch.fnmatch(snapfile, f'{SnapshotFileBase}_*.hdf5') and
            os.path.isfile(f'{SnapshotPath}/{snapfile}')):

          snap = int(snapfile.split('.hdf5')[0].split('_')[1])
          if snap > lastsnap:
            lastsnap = snap

    foundSnap = False
    for isnap in range(snapnum+1, lastsnap+1):

      if hasSubdir:
        snapfile = f'{SnapshotPath}/{SnapshotFileBase}_{isnap:04}/' + \
                   f'{SnapshotFileBase}_{isnap:04}.hdf5'
      else:
        snapfile = f'{SnapshotPath}/{SnapshotFileBase}_{isnap:04}.hdf5'

      nextsnap = h5py.File(snapfile, 'r')

      if nextsnap['Header'].attrs['SelectOutput'].decode() == 'Snipshot':
        # Looking for snapshot, not snipshot
        nextsnap.close()
        continue

      print(f'Getting stellar ages from snapshot {isnap}')
      foundSnap = True
      break

    if foundSnap:
      # Ok, found a snapshot. Now get the ages
      Future_IDs = nextsnap[f'PartType{istar}/ParticleIDs'][:]
      Future_Ages = load_physical_data(
          nextsnap[f'PartType{istar}'], 'Ages', ascale)

      # Correct future ages to current time
      futureTime = nextsnap['Header'].attrs['Time'][0]
      Future_Ages -= (futureTime - simTime)

      _, snip_ind, snap_ind = np.intersect1d(Stars_IDs, Future_IDs,
          assume_unique=True, return_indices=True)

      Stars_Ages = np.zeros(len(Stars_IDs), dtype=Future_Ages.dtype)
      Stars_Ages[snip_ind] = Future_Ages[snap_ind]

      # Also need to store Age attrs for later
      clusters['Age_attrs'] = {}
      for key in nextsnap['PartType4/Ages'].attrs.keys():
        clusters['Age_attrs'][key] = nextsnap['PartType4/Ages'].attrs[key]

      nextsnap.close()

    else:
      print("Couldn't find a future snapshot for stellar ages, exiting...")
      exit()


  timing['I/O'] += (time.time() - start) / 60.

  current, peak = tracemalloc.get_traced_memory()
  print(f'Current memory usage: {current/1024.**3:.5g} GB')


  timing_haloes = np.zeros(len(subhaloes), dtype=np.float32)

  def df_one_subhalo(isub):
    '''
    Set up dynamical friction calculation for one halo to parallelise over
    '''

    # Timing
    start_halo = time.time()
    start = time.time()
    thalo_reproc = 0.
    thalo_io = 0.
    thalo_menc = 0.
    thalo_ecc = 0.
    thalo_df = 0.

    Nbound = subhaloes[isub]['Nbound']
    Ntype = subhaloes[isub]['NboundType']

    # Only resolved haloes with stars
    if (Nbound == 0) or (Ntype[istar] == 0):
      return

    # Allocate space for subhalo particle properties
    SH_type = np.zeros(Nbound, dtype=int)
    SH_masses = np.zeros(Nbound)
    SH_pos = np.zeros((Nbound, 3))
    SH_vel = np.zeros((Ntype[iDM]+Ntype[istar], 3))
    SH_Star_IDs = np.array([], dtype=int)
    if snapPotentials:
      SH_pots = np.zeros(Nbound)

    thalo_reproc += time.time() - start

    # Match the subhalo particles and get a list of all particle properties
    if loadall:
      # Data already loaded so just match particles
      start = time.time()

      upto = 0
      upto_vel = 0
      for ptype in ['Gas', 'DM', 'Stars', 'BH']:
        ipart = itypes[ptype]

        if Ntype[ipart] == 0: continue

        # Match the snapshot and subhalo particles
        _, p_ind, sub_ind = np.intersect1d(All_IDs[ptype], subparts[isub],
            assume_unique=True, return_indices=True)

        SH_type[upto:upto+Ntype[ipart]] = ipart * np.ones(Ntype[ipart], dtype=int)
        SH_masses[upto:upto+Ntype[ipart]] = All_masses[ptype][p_ind]
        SH_pos[upto:upto+Ntype[ipart]] = All_pos[ptype][p_ind]
        if snapPotentials:
          SH_pots[upto:upto+Ntype[ipart]] = All_pots[ptype][p_ind]

        if ptype in ['DM', 'Stars']:
          SH_vel[upto_vel:upto_vel+Ntype[ipart]] = \
              All_vel[ptype][p_ind]
          upto_vel += Ntype[ipart]

        if ptype == 'Stars':
          SH_Star_IDs = All_IDs[ptype][p_ind]

        upto += Ntype[ipart]

      thalo_reproc += time.time() - start

    else:
      # Load particles for this halo from file

      upto = 0
      upto_vel = 0
      for ptype in ['Gas', 'DM', 'Stars', 'BH']:

        ipart = itypes[ptype]
        if Ntype[ipart] == 0: continue

        start = time.time()

        group = snapshot[f'PartType{ipart}']

        # Match the snapshot and subhalo particles
        _, p_ind, _ = np.intersect1d(All_IDs[ipart], subparts[isub],
            assume_unique=True, return_indices=True)

        thalo_reproc += time.time() - start
        start = time.time()

        if ptype == 'BH':
          MassType = 'DynamicalMasses'
        else:
          MassType = 'Masses'
 
        SH_type[upto:upto+Ntype[ipart]] = ipart * np.ones(Ntype[ipart], dtype=int)
        SH_masses[upto:upto+Ntype[ipart]] = \
            load_physical_data(group, MassType, ascale, p_ind)
        SH_pos[upto:upto+Ntype[ipart]] = \
            load_physical_data(group, 'Coordinates', ascale, p_ind)
        if snapPotentials:
          SH_pots[upto:upto+Ntype[ipart]] = \
              load_physical_data(group, 'Potentials', ascale, p_ind)
 
        if ptype in ['DM', 'Stars']:
          SH_vel[upto_vel:upto_vel+Ntype[ipart]] = \
              load_physical_data(group, 'Velocities', ascale, p_ind)
          upto_vel += Ntype[ipart]

        if ptype == 'Stars':
          SH_Star_IDs = np.array(All_IDs[ipart][p_ind], copy=True)
 
        upto += Ntype[ipart]
 
        thalo_io += time.time() - start
 
    start = time.time()
 
    # Match the snapshot particles to subhalo particles
    _, s_ind, sub_ind = np.intersect1d(Stars_IDs, SH_Star_IDs,
        assume_unique=True, return_indices=True)
 
    clusters['dfTrackId'][s_ind] = subhaloes['TrackId'][isub]
 
    # Does this subhalo have any clusters?
    if np.sum(GCs_Masses[s_ind]) == 0: return

    if snapPotentials:
      # Use particle snapshot potentials for centre of potential
      cofp = SH_pos[ SH_pots.argmin() ]
    else:
      # Use subhalo catalogue for centres
      cofp = subhaloes['ComovingMostBoundPosition'][isub] * ascale
 
    # centre positions and velocities on subhalo centre of potential
    SH_pos = centre_periodic_positions(SH_pos, cofp, boxsize)
    SH_vel -= subhaloes['PhysicalMostBoundVelocity'][isub]
 
    SH_radii = np.linalg.norm(SH_pos, axis=1)
 
    # Radii of the stars
    s_mask = (SH_type == istar)
 
    s_radii = SH_radii[s_mask][sub_ind]
 
    # Prepare for the DF timescale calculations
 
    # Sort subhalo particles by radius
    # We assume a spherically symmetric potential, which is probably ok for
    # a DM-dominated system
 
    # Particles with velocities first
    vel_mask = np.bitwise_or(SH_type == iDM, SH_type == istar)
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
 
    thalo_reproc += time.time() - start
    start = time.time()
 
    # Enclosed mass
    menc = np.zeros(len(SH_masses))
    menc[0] = 0.
    for i in range(1, len(menc)):
      menc[i] = menc[i-1] + SH_masses[i]
 
    # Approx. upper limit to the distance a cluster can inspiral from
    # Used to limit to calculations worth doing
    Mtest = 1.1 * np.max(GCs_Masses[s_ind])
    Rmax = SH_radii[-1]
    for i in range(len(menc)):
      # Just check stars
      if SH_type[i] != istar: continue

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
    do_stars = np.zeros(len(s_ind), dtype=bool)
    for i in range(len(s_ind)):
 
      # Too far, don't bother
      if s_radii[i] > Rmax: continue
 
      # Does this particle have any clusters?
      if np.sum(GCs_Masses[s_ind[i]]) == 0: continue
 
      if s_radii[i] == 0:
        # Can stop here. t_df = 0 for r = 0
        for j in range( gc_shape[1] ):
          if GCs_Masses[s_ind[i],j] > 0:
            clusters['tfric'][s_ind[i], j] = 0.

      else:
        do_stars[i] = True
 
    thalo_reproc += time.time() - start
    start = time.time()
 
    if np.sum(do_stars) == 0: return
 
    # Outer component of potential, assuming spherical symmetry
    # -G int_r^inf dM(r')/r'
    potential_outer = np.zeros(len(SH_masses))
    for i in range(len(SH_masses)-2,-1,-1):
      potential_outer[i] = \
          potential_outer[i+1] - const_G * SH_masses[i+1] / SH_radii[i+1]
 
    # get eccentricities for all stars at once
    dx = centre_periodic_positions(
        Stars_Pos[s_ind], cofp, boxsize)
    dv = Stars_Vel[s_ind] - subhaloes['PhysicalMostBoundVelocity'][isub]
 
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
    for i in range(len(s_ind)):
 
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
      part_GC_masses = GCs_Masses[s_ind[i]]
      for j in range( gc_shape[1] ):
        if part_GC_masses[j] <= 0: continue

        if Mgal == 0:
          clusters['tfric'][s_ind[i], j] = 0.

        else:
          clusters['tfric'][s_ind[i], j] = get_df_timescale(
              Mgal, part_GC_masses[j], rcirc[i], epsilon[i], velStdv, const_G)

    thalo_df += time.time() - start
    end_halo = time.time()

    with timing_lock:
      timing['TotalHaloes'] += (end_halo - start_halo) / 60.
      timing['Reprocessing'] += thalo_reproc / 60.
      timing['I/O'] += thalo_io / 60.
      timing['EnclosedMasses'] += thalo_menc / 60.
      timing['Eccentricities'] += thalo_ecc / 60.
      timing['DFCalculation'] += thalo_df / 60.
      timing_haloes[isub] = (end_halo - start_halo) / 60.

    return

  # Loop over all the subhaloes, sorting to do largest haloes first
  submask = subhaloes['NboundType'][:,istar] > 0
  nsort = subhaloes['Nbound'][submask].argsort()[::-1]
  do_halo = np.arange(len(subhaloes))[submask][nsort]

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

  # If dynamical friction timescale is shorter than cluster age then cluster 
  # is removed/considered disrupted
  # Assumes particle has been in the same halo the whole time. DF time will
  # generally significantly increase if the host galaxy is accreted, so should
  # mainly only affect the in situ GCs
  GC_ages = (np.ones(gc_shape).T * np.array(Stars_Ages)).T

  df_mask = (clusters['tfric'] < GC_ages)
  clusters['removed'][df_mask] = 1
  clusters['snapNumRemoved'][df_mask] = snapnum
  clusters['massRemoved'][df_mask] = np.array(GCs_Masses)[df_mask]
  clusters['ageRemoved'][df_mask] = GC_ages[df_mask]

  mask = GCs_Masses > 0
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

def write_clusters(clusters, snapshot, output, snapnum):
  '''
  Write dynamical friction data to HDF5 file

  Parameters
  ----------
  clusters : The clusters dictionary
  snapshot : HDF5 snapshot file
  output : Output path
  snapnum : Snapshot number
  '''

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

  ds = group.create_dataset('GCs_DF_TrackId', data=clusters['dfTrackId'],
      dtype='i4', compression="gzip")
  for key in snapshot['PartType4/ParticleIDs'].attrs.keys():
    ds.attrs[key] = snapshot['PartType4/ParticleIDs'].attrs[key]
  ds.attrs['Description'] = "HBT+ TrackId of halo in which GC resides"

  f_df.close()
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
    dest='basename', default='snapshot',
    help="Snapshot basename (default 'snapshot')")
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
  parser.add_argument('--loadall', action='store_true',
    dest='loadall', default=False,
    help="Try and load all data at once to speed up I/O. Otherwise load "
         "halo particles individually")
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
  print('Loadall:', args.loadall)
  print('Snap potentials:', args.snapPotentials)
  sys.stdout.flush()

  # Track memory usage
  tracemalloc.start()

  # Make the output directory if it doesn't exist yet
  os.system(f'mkdir -pv {args.output}')

  # Load from snapshot. First check for a subdir
  snapfile = f'{args.path}/{args.basename}_{args.snapnum:04}/' + \
             f'{args.basename}_{args.snapnum:04}.hdf5'

  if not os.path.isfile(snapfile):
    snapfile = f'{args.path}/{args.basename}_{args.snapnum:04}.hdf5'

  print('Loading snapshot from', snapfile)
  sys.stdout.flush()
  snapshot = h5py.File(snapfile, 'r')

  if args.skipSnips and \
      (snapshot['Header'].attrs['SelectOutput'].decode() == 'Snipshot'):
    print('Snipshot, exiting...')
    snapshot.close()
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
  clusters = get_removed_clusters(snapshot, args, numthreads)
  print('Dynamical friction took %.5g mins' % ((time.time() - start)/60.))

  print(f'Writing output to {args.output}/df_timescale_{args.snapnum:04}.hdf5')
  sys.stdout.flush()
  write_clusters(clusters, snapshot, args.output, args.snapnum)

  snapshot.close()

  current, peak = tracemalloc.get_traced_memory()
  print(f'Peak memory usage: {peak/1024.**3:.5g} GB')
  tracemalloc.stop()

