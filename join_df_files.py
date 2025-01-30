import sys, os, numpy as np, h5py
import fnmatch
import argparse


if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--path', action='store', type=str,
    dest='path', default='dynamical_friction',
    help="Path to dynamical friction files (default 'dynamical_friction')")
  parser.add_argument('--snaplist', action='store', type=str,
    dest='snaplist', default='',
    help="List of comma-separated snapshot indicies. " +
    "Otherwise take maximum index from maxsnap.")
  parser.add_argument('maxsnap', nargs='?', action='store', type=int,
    default=-1, help="Maximum snapshot index")
  args = parser.parse_args()

  print('*** joining DF snapshots ***')
  print()
  print('pid:', os.getpid())
  print('path:', args.path)

  # Specity a list, or the maximum of a range
  if len(args.snaplist) > 0:
    snaplist = np.fromstring(args.snaplist, dtype=int, sep=',')

  else:
    print('MaxSnapshotIndex:', args.maxsnap)
    snaplist = np.arange(args.maxsnap+1)

  print('Running...')

  for isnap in range(1, len(snaplist)):
    snap = snaplist[isnap]

    print(f'Snapshot {snap}')
    sys.stdout.flush()

    # The previous snapshot
    lastsnap = snaplist[isnap-1]
    try:
      f_prev = h5py.File(f'{args.path}/df_timescale_{lastsnap:04}.hdf5', "r")
    except: 
      print(f"df_timescale_{lastsnap:04}.hdf5 doesn't exist")
      continue

    # If there's no stars yet we can stop here
    PartTypeNames = f_prev['Header/PartTypeNames'][:]
    for i in range(len(PartTypeNames)):
      if PartTypeNames[i].decode() == 'Stars':
        if (f_prev['Header'].attrs['NumPart_Total'][i] == 0):
          print( 'No stars in snapshot')
          f_prev.close()
          continue

    ParticleIDs_prev = f_prev['/PartType4/ParticleIDs'][:]
    Removed_prev = f_prev['/PartType4/GCs_DF_Removed'][:]
    SnapnumRemoved_prev = f_prev['/PartType4/GCs_DF_SnapnumRemoved'][:]
    Timescale_prev = f_prev['/PartType4/GCs_DF_Timescale'][:]
    AgeRemoved_prev = f_prev['/PartType4/GCs_DF_AgeRemoved'][:]
    MassRemoved_prev = f_prev['/PartType4/GCs_DF_MassRemoved'][:]
    TrackId_prev = f_prev['/PartType4/GCs_DF_TrackId'][:]
    f_prev.close()

    # The current snapshot, which we'll update in place
    f = h5py.File(f'{args.path}/df_timescale_{snap:04}.hdf5', "r+")
    ParticleIDs = f['/PartType4/ParticleIDs'][:]
    Removed = f['/PartType4/GCs_DF_Removed'][:]
    SnapnumRemoved = f['/PartType4/GCs_DF_SnapnumRemoved'][:]
    Timescale = f['/PartType4/GCs_DF_Timescale'][:]
    AgeRemoved = f['/PartType4/GCs_DF_AgeRemoved'][:]
    MassRemoved = f['/PartType4/GCs_DF_MassRemoved'][:]
    TrackId = f['/PartType4/GCs_DF_TrackId'][:]

    print('  removed prev',np.sum(Removed_prev))
    print('  removed',np.sum(Removed))
    sys.stdout.flush()

    # Link between previous and current snapshots
    _, prev_ind, new_ind = np.intersect1d(ParticleIDs_prev, ParticleIDs,
        assume_unique=True, return_indices=True)

    GC_shape = np.shape(Removed)
 
    # Now copy the data
    for i in range(len(ParticleIDs_prev)):

      prev_i = prev_ind[i]
      new_i = new_ind[i]

      # GCs within particle retain same order
      for j in range(GC_shape[1]):

        if Removed_prev[prev_i,j]:
          # Was flagged as removed in last snapshot, so copy through data
          Removed[new_i,j] = Removed_prev[prev_i,j]
          SnapnumRemoved[new_i,j] = SnapnumRemoved_prev[prev_i,j]
          Timescale[new_i,j] = Timescale_prev[prev_i,j]
          AgeRemoved[new_i,j] = AgeRemoved_prev[prev_i,j]
          MassRemoved[new_i,j] = MassRemoved_prev[prev_i,j]
          TrackId[new_i,j] = TrackId_prev[prev_i,j]
 
    # write back to hdf5 file
    dataset = f['/PartType4/ParticleIDs']
    dataset[...] = ParticleIDs
 
    dataset = f['/PartType4/GCs_DF_Removed']
    dataset[...] = Removed
 
    dataset = f['/PartType4/GCs_DF_SnapnumRemoved']
    dataset[...] = SnapnumRemoved
 
    dataset = f['/PartType4/GCs_DF_Timescale']
    dataset[...] = Timescale
 
    dataset = f['/PartType4/GCs_DF_AgeRemoved']
    dataset[...] = AgeRemoved
 
    dataset = f['/PartType4/GCs_DF_MassRemoved']
    dataset[...] = MassRemoved
 
    dataset = f['/PartType4/GCs_DF_TrackId']
    dataset[...] = TrackId
 
    f.close()

  print('Done')
