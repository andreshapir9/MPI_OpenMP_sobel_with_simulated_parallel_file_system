This program uses MPI to deploy between 1 and N processes across an equal amount of nodes.
Then it uses OpenMP to speed up Sobel edge detection across 4 threads.
 

Simulates a parallel file system by using the /tmp directory.

MPI allows splitting the work across processes so I/O can be sped up
OpenMP allows the sobel calculation to be sped up too


Host file : c2_hosts was designed for the California State University Chico cluster, but can be modified to work with any cluster.
req: must have SSH access to all nodes without password
req: must populate /tmp/sg_input_images with images to be processed on all nodes
req: must create /tmp/output_files on all nodes
Req: The Images must be in .pgm format


 To Compile
- Make
    - To run Single-threaded
    - Change the THREADS value to 1
    - Set read_file_name folder_name to sg_input_images
    - Make sure /tmp/sg_input_images exists and its populated, make sure /tmp/output_files exists
    - mpirun mpirun -n 1 -ppn 1 -f c2_hosts ./sobel_MPI_scalable
- To Run OpenMP
    - Change the THREADS value to 2 or 4
    - Set read_file_name folder_name to sg_input_images
    - Make sure /tmp/sg_input_images exists and its populated, make sure /tmp/output_files exists through all
    - mpirun mpirun -n 1 -ppn 1 -f c2_hosts ./sobel_MPI_scalable
- To run MPI
    - Change the THREADS value to 1
    - Set read_file_name folder_name to mp_input_images
    - Make sure /tmp/mp_input_images exists and its populated, make sure /tmp/output_files exists through all
    - mpirun mpirun -n 1(number_of_processes) -ppn 1 -f c2_hosts ./sobel_MPI_scalable
- To run hybrid
    - Change the THREADS value to 2 or 4
    - Set read_file_name folder_name to mp_input_images
    - Make sure /tmp/mp_input_images exists and its populated, make sure /tmp/output_files exists through all
    - mpirun mpirun -n 1(number_of_processes) -ppn 1 -f c2_hosts ./sobel_MPI_scalable
    