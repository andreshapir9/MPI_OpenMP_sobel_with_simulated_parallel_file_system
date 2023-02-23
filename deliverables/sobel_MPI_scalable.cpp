/*
    sources used to create this code:
        https://github.com/sorazy/canny/blob/master/sobel.cpp
        https://stackoverflow.com/questions/20961470/openmp-c-sobel-edge-detection


    WARNING:
        reads from /tmp
        meaning folders might have to be created before running
        if folder is not found code will just code will exit


    compile with:
        mpicxx -g -O3 -Wall  -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/ -fopenmp sobel_MPI_scalable.cpp -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
        or given makefile
            make
    execute with( num_processes = number of nodes who have folder in /tmp)
        mpirun  mpirun -n (num_processes)  -ppn 1 -f c2_hosts ./sobel_MPI_scalable

    Output:
        prints out the time it took to process each image
        prints out the average time it took to process all images
        prints out the standard deviation of the time it took to process all images
        prints out the total time it took to process all images

    NOTE:
        This program is simulates a parallel file system
        meaning that each node has a folder with the same name
        and the output is written to to different folders all located in /tmp
        inside each devices /tmp folder

        the input folder names are:
            sg_input_images
            mp_input_images
        the output folder names are:
            output_images
    
    NOTE:
        To run as single threaded change the #define THREADS 4 to #define THREADS 1
        and set the number of processes to 1 in the command line

    
    NOTE:
        to set files or get files from the nodes use the following commands
            sftp node_name:/tmp/folder_name/
                get file_name
                put file_name
*/

#include <cmath>
#include <fstream>
#include <iostream>
#include <climits>
#include <mpi.h>
#include<omp.h>
#include<vector>
#include <dirent.h>

using namespace std;

//mask to find horizontal edges
int maskx[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};

//mask to find vertical edges
int masky[3][3] = {{1,2,1}, {0,0,0}, {-1,-2,-1}};

#define THREADS 4
#define THRESHOLD 35

int thread_rank,size;

//enumeration to keep track of read file
enum read_file_name{
    sg_input_images,
    mp_input_images
};

int main(int argc, char **argv)
{
    //variables to keep track of time
    double start, end;
    read_file_name folder_name = mp_input_images;
    //MPI initialization
    MPI_Init(&argc, &argv); //initialize MPI operations
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_rank); //get the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes
    start = MPI_Wtime();



    vector<double> sobel_start_times, sobel_end_times;

    //vector of image names to be processed
    vector<string> pictures;

    //image information
    string header;
    int width, height, intensity;


    //open the folder and read all of the file names we use the enum to determine which folder to open
    DIR *dr;
    struct dirent *ent;
    string directory = "/tmp/";
    if(folder_name == sg_input_images)
        directory += "sg_input_images/";
    else if(folder_name == mp_input_images)
        directory += "mp_input_images/";
    else{
        cout<<"Invalid folder name"<<endl;
        exit(-1);
    }
    if((dr = opendir(directory.c_str())) != NULL){
        while((ent = readdir(dr)) != NULL){
            //we check if the file is a .pgm file
            string file_name = ent->d_name;
            if(file_name.find(".pgm") != string::npos){
                pictures.push_back(file_name);
              //  cout << file_name << endl;
            }
        }
        closedir(dr);
    }
    else{
        cout<<"Could not open folder: "<<directory<<endl;
        return 0;
    }

    //we check that we have at least one file
    if(pictures.size() == 0)
    {
        cout<<"No files in directory"<<endl;
        exit(-1);
    }

    //we process each image
    for(int i = 0; i < pictures.size(); i++){

        //we open the image
        string file_name = directory + pictures[i];
        ifstream file(file_name.c_str(), ios::in | ios::binary);
        if(!file.is_open()){
            cout<<"Could not open file: "<<file_name<<endl;
            exit(-1);
        }

        //we read the header and image information
        file >> header;
        if(header != "P5"){
            cout<<"Not a PGM file"<<endl;
            exit(-1);
        }
        file >> width >> height >> intensity;
       // cout << "width: " << width << " height: " << height << " intensity: " << intensity << endl;

        //using image information we create the image arrays
        double image[height][width];//image to be processed
        double imageX[height][width];//image to store horizontal edges
        double imageY[height][width];//image to store vertical edges
        double magnitude[height][width]; //image to store the magnitude

        //we read the image
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                image[i][j] = (int)file.get();
            }
        }
        //close the file
        file.close();


        //time before
        sobel_start_times.push_back(MPI_Wtime());

        //we apply the sobel filter on each 3x3 pixel group
#pragma omp parallel for num_threads(THREADS)
        for(int i = 1; i < height-1; i++){
            for(int j = 1; j < width-1; j++){

                //we apply the mask to the image and keep track of the sum
                double sumX = 0, sumY = 0;
                for(int k = -1; k <= 1; k++){
                    for(int l = -1; l <= 1; l++){
                        sumX += image[i+k][j+l] * maskx[k+1][l+1];
                        sumY += image[i+k][j+l] * masky[k+1][l+1];
                    }
                }

                //we store the sum in the imageX and imageY arrays(we could just store the sum in the magnitude array)
                //gives us flexibility to use the imageX and imageY arrays for other things if we want
                imageX[i][j] = sumX;
                imageY[i][j] = sumY;

                //we calculate the magnitude if the sum is greater than a threshold we set it to 255 else we set it to 0
                int res= (int)sqrt(abs(sumX*sumX) + abs(sumY*sumY));
                if(res > THRESHOLD)
                    magnitude[i][j] = 255;
                else
                    magnitude[i][j] = 0;

            }
        }

        //time after sobel
        sobel_end_times.push_back(MPI_Wtime());

        //we write the magnitude to a file
        string file_name2 = "/tmp/output_files/" + pictures[i]+"_magnitude.pgm";
        ofstream file2(file_name2.c_str(), ios::out | ios::binary|ios::trunc);
        if(!file2.is_open()){
            cout<<"Could not open file: "<<file_name2<<endl;
            exit(-1);
        }
        file2 << header << endl;
        file2 << width << " " << height << endl;
        file2 << intensity << endl;

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                file2 << (char)magnitude[i][j];
            }
        }

        file2.close();

        //one whole itteration is done
        
        
    }

    //we finish timing
    end = MPI_Wtime();
    

    //calculate some stats on the sobel times
    double time = end - start;
    double max_time, min_time, avg_time;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(thread_rank == 0){
        cout << "Max time: " << max_time << endl;
        cout << "Min time: " << min_time << endl;
        cout << "Avg time: " << avg_time/size << endl;
        cout << "time per image: " << avg_time/size/pictures.size() << " on " << size << " threads each with " << pictures.size() << " images" << endl;
        if(size > 1){
            cout << " min time per image: " << min_time/pictures.size() << endl;
            cout << " max time per image: " << max_time/pictures.size() << endl;
        }
    }
    //lets calculate how long we spent in sobel
    double sobel_time = 0;
    for(int i = 0; i < sobel_start_times.size(); i++){
        sobel_time += sobel_end_times[i] - sobel_start_times[i];
    }
    double max_sobel_time, min_sobel_time, avg_sobel_time;
    MPI_Reduce(&sobel_time, &max_sobel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sobel_time, &min_sobel_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sobel_time, &avg_sobel_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(thread_rank == 0){
        cout << "Max sobel percent: " << max_sobel_time/max_time*100 << endl;
        cout << "Min sobel percent %: " << min_sobel_time/min_time*100 << endl;
        cout << "Avg sobel percent %: " << avg_sobel_time/(avg_time)*100 << endl;
    }

    //calculate the true end of the program
    end = MPI_Wtime();
    if(thread_rank == 0){
        cout << "Total time: " << end - start << endl;
    }
    MPI_Finalize();
}


