#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

std::string strindex(int beg, int end, std::string line){
    int len = line.length();
    return line.substr(beg, len - beg - end);
}


int main() {
    int nblock = 0, nexp = 0, nfeatures = 10, dataindex = 0;
    bool firstB = true;
    std::ifstream FILE("temp_file.txt");
    std::string line, blocksize_row, blocksize_col, threads, Nflops;
    std::vector<std::vector<std::string>> data(1, std::vector<std::string>(nfeatures));
    std::vector<std::string> features(nfeatures);
    data[0][0] = "Nthr";
    data[0][1] = "Ms";
    data[0][2] = "Mflop";
    data[0][3] = "Nite";
    data[0][4] = "Brow";
    data[0][5] = "Bcol";
    data[0][6] = "t1";
    data[0][7] = "t2";
    data[0][8] = "d1";
    data[0][9] = "d2";
    
    std::getline(FILE, line); 
    
    while(std::getline(FILE, line)){
       // std::cout << line << std::endl;
        
        //checking if benchmark
        if(line.empty()){
            
            std::getline(FILE, line);
            std::getline(FILE, line);
            do
            {
                std::getline(FILE, line);
                //read Nite and Nflops
                if (line == "done") { break; } 
                dataindex += 1;
                //std::cout <<dataindex << std::endl;
              
                line = strindex(5, 0, line);
                std::stringstream ss(line);
                
                for( int i(1); i < nfeatures; i++){
                    std::getline(ss, features[i], ' ');
                  //  std::cout << ";" << features [i];
                }
                
                std::getline(ss, Nflops, ' ');
                //std::cout << " " <<  Mflops << std::endl;

                // put in data accordingly 
                if (firstB) {
                    nexp += 1;
                    data.push_back(std::vector<std::string>(nfeatures + 1));
                    data[dataindex][0] = threads;
                    for( int i(1); i < nfeatures; i++){
                        data[dataindex][i] = features[i];
                    }
                    data[dataindex][nfeatures] = Nflops;
                }
                else { data[dataindex].push_back(Nflops); }
       
            } while(true);
        }

        //checking if its chunk size update
        else if(line[0] == '#'){
            dataindex = 0;
            nblock += 1;
            if(nblock == 2){ firstB = false;}
            blocksize_row = strindex(10, 4, line);
            
            std::getline(FILE, line);
            blocksize_col = strindex(10, 4, line);
            
            
            data[0].push_back(blocksize_row + 'X' + blocksize_col);




        }
        //checking if its tread update
        else if(line[0] == '-'){
            threads = strindex(4, 12, line);
        }


    }



    std::ofstream OUT("data.dat");
    OUT << nexp << " " << nfeatures << " " << nblock << std::endl;
    for( int i(0); i < data.size(); i++) {
        for (int j(0); j < data[0].size(); j++) {
            OUT << data[i][j] << " ";
        }
        OUT << "\n";
    }
    


    return 0;
}
