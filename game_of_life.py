import numpy
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
stat = MPI.Status()

# Set global variables
#COLS = 400
#ROWS = 198
COLS = 20
ROWS = 10

if size > ROWS:
        print("Not enough ROWS")
        exit()
subROWS=ROWS//size+2

if size > ROWS:
        print("Not enough ROWS")
        exit()

def msgUp(subGrid):
        # Sends and Recvs rows with Rank+1
        comm.send(subGrid[subROWS-2,:],dest=rank+1)
        subGrid[subROWS-1,:]=comm.recv(source=rank+1)
        return 0

def msgDn(subGrid):
        # Sends and Recvs rows with Rank-1
        comm.send(subGrid[1,:],dest=rank-1)
        subGrid[0,:] = comm.recv(source=rank-1)
        return 0

def computeGridPoints(subGrid):
        interSubGrid = numpy.copy(subGrid)
        for subROW in range(1,subROWS-1):
                for COLelem in range(1,COLS-1):
                        sum = ( subGrid[subROW-1,COLelem-1]+subGrid[subROW-1,COLelem]+subGrid[subROW-1,COLelem+1]
                +subGrid[subROW,COLelem-1]+subGrid[subROW,COLelem+1]
                +subGrid[subROW+1,COLelem-1]+subGrid[subROW+1,COLelem]+subGrid[subROW+1,COLelem+1] )
                        if subGrid[subROW,COLelem] == 1:
                                if sum < 2:
                                        interSubGrid[subROW,COLelem] = 0
                                elif sum > 3:
                                        interSubGrid[subROW,COLelem] = 0
                                else:
                                        interSubGrid[subROW,COLelem] = 1
                        if subGrid[subROW,COLelem] == 0:
                                if sum == 3:
                                        interSubGrid[subROW,COLelem] = 1
                                else:
                                        interSubGrid[subROW,COLelem] = 0
        subGrid = numpy.copy(interSubGrid)
        return 0


prob = 0.7
N=numpy.random.binomial(1,prob,size=subROWS*COLS)
subGrid=numpy.reshape(N,(subROWS,COLS))

if rank == 0:
    subGrid[0,:] = 0
if rank == size-1:
    subGrid[subROWS-1,:] = 0
subGrid[:,0] = 0
subGrid[:,COLS-1] = 0

print("First Generation")


# The main body of the algorithm
#compute new grid and pass rows to neighbors
generations = 6

for i in range(generations):
        computeGridPoints(subGrid)
        #exhange edge rows for next interation
        if rank == 0:
                msgUp(subGrid)
        elif rank == size-1:
                msgDn(subGrid)
        else:
                msgUp(subGrid)
                msgDn(subGrid)


Grid=comm.gather(subGrid[1:subROWS-1,:],root=0)

if rank == 0:
        result= numpy.vstack(Grid)
#       print numpy.vstack(initGrid)
        print(result[:])
#       print len(result)
