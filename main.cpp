#define BUFFER_SIZE 20
#define CORR_BORDER 0.99

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <math.h>
#include <time.h>
#include <barrier>

#include "Eigen/Dense"

class GMDHSolver{
        public:
        GMDHSolver(int matrix_height, int matrix_width,int buffer_size, int number_of_iterations, int amount_of_threads);
        ~GMDHSolver();
        std::vector<Eigen::VectorXf> predict(Eigen::MatrixXf X);
        Eigen::MatrixXf A;
        Eigen::VectorXf b;
        int amount_of_pairs;
        int amount_of_threads;
        int matrix_height;
        int matrix_width;
        int buffer_size;
        int number_of_iterations;
        std::mutex mutex;
        std::barrier<>* barrier;
        std::vector<std::thread> threads; 
        std::vector<std::vector<Eigen::VectorXf>> thread_buffers;
        std::vector<std::vector<std::pair<int,int>>> thread_index_buffers;
        std::vector<std::vector<std::pair<int,int>>> result_index_buffer;
        std::vector<std::vector<Eigen::VectorXf>> result_buffer;
        void solver(int thread_number);
        float correlation(Eigen::VectorXf x, Eigen::VectorXf y);
        float r2_score(Eigen::VectorXf pred, Eigen::VectorXf y);
        
        //Utils for vectors
        template<class T>
        float max(std::vector<T> vector);
        template<class T>
        float min(std::vector<T> vector);
        template<class T>
        std::pair<float,int> find_min(std::vector<T> vector);
};
//Корреляция
float GMDHSolver::correlation(Eigen::VectorXf x, Eigen::VectorXf y){
        float up = x.dot(y);
        float down = sqrt(x.dot(x) + y.dot(y));
        return up / down;
}
//Матрика R-квадрат
float GMDHSolver::r2_score(Eigen::VectorXf pred, Eigen::VectorXf y){
        float sum_squares = (y - (y.mean() * Eigen::VectorXf::Ones(matrix_height))).squaredNorm();
        float sum_residuals_squared = (y - pred).squaredNorm();
        return 1 - (sum_residuals_squared / sum_squares);
}

GMDHSolver::GMDHSolver(int matrix_height, int matrix_width,int buffer_size,int number_of_iterations, int amount_of_threads){
        this->amount_of_threads = amount_of_threads;
        this->matrix_height = matrix_height;
        this->matrix_width = matrix_width;
        this->buffer_size = buffer_size;
        this->amount_of_pairs = (matrix_width * (matrix_width - 1)) / 2;
        this->number_of_iterations = number_of_iterations;
        this->barrier = new std::barrier<>(amount_of_threads);
        this->thread_buffers = std::vector<std::vector<Eigen::VectorXf>>(amount_of_threads);
        this->thread_index_buffers = std::vector<std::vector<std::pair<int,int>>>(amount_of_threads);
        A = Eigen::MatrixXf::Random(matrix_height,matrix_width);
        b = Eigen::VectorXf::Random(matrix_height);
        //Запускаем потоки...
        for(int i = 0; i < amount_of_threads - 1; i++){
                threads.push_back(std::thread(&GMDHSolver::solver,this,i));
        }
        solver(amount_of_threads - 1);
        //... и ждем их завершения, только потом выходим.
        for(int i = 0; i < amount_of_threads - 1; i++){
                threads[i].join();
        }
}

GMDHSolver::~GMDHSolver(){
        
}

template<class T>
float GMDHSolver::max(std::vector<T> vector){
        float max = vector[0];
        for(int i = 0; i < vector.size(); i++){
                if(vector[i] > max) max = vector[i];
        }
        return max;
}
template<class T>
float GMDHSolver::min(std::vector<T> vector){
        float min = vector[0];
        for(int i = 0; i < vector.size(); i++){
                if(vector[i] < min) min = vector[i];
        }
        return min;
}
template<class T>
std::pair<float,int> GMDHSolver::find_min(std::vector<T> vector){
        std::pair<float,int> result;
        result.first = vector[0];
        result.second = 0;
        for(int i = 0; i < vector.size(); i++){
                if(vector[i] < result.first) 
                {
                result.first = vector[i];
                result.second = i;
                }
        }
        return result;
}


void GMDHSolver::solver(int thread_number){
        std::vector<std::vector<Eigen::VectorXf>> buffer(number_of_iterations);
        std::vector<std::vector<std::pair<int,int>>> index_buffer(number_of_iterations);
        std::vector<float> buffer_r2;
        std::vector<float> buffer_correlation;
        Eigen::MatrixXf A_local(matrix_height,2);
        Eigen::VectorXf solution;
        Eigen::VectorXf prediction;

        std::pair<float,int> min_val_pos;

        //Primary iteration
        
        for(int i = 0; i < matrix_width; i++){
                A_local.col(0) = A.col(i);
                for(int j = i + thread_number; j < matrix_width; j += amount_of_threads){
                        A_local.col(1) = A.col(j);
                        solution = A_local.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                        prediction = A_local * solution;
                        if(buffer[0].size() == 0){
                                buffer[0].push_back(prediction);
                                index_buffer[0].push_back(std::pair<int,int>(i,j));
                        } else {
                                for(int k = 0; k < buffer[0].size(); k++){
                                        buffer_correlation.push_back(correlation(prediction,buffer[0][k]));
                                }
                                if(buffer[0].size() < BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__){
                                        buffer[0].push_back(prediction);
                                        index_buffer[0].push_back(std::pair<int,int>(i,j));
                                } else if (buffer[0].size() >= BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__){
                                        for(int k = 0; k < buffer[0].size(); k++){
                                                buffer_r2.push_back(r2_score(b,buffer[0][k]));
                                        }
                                        if(r2_score(b,prediction) > min<float>(buffer_r2)){
                                                min_val_pos = find_min<float>(buffer_r2);
                                                buffer[0][min_val_pos.second] = prediction;
                                                index_buffer[0][min_val_pos.second] = std::pair<int,int>(i,j);
                                        }
                                }
                        }
                        buffer_correlation.clear();
                        buffer_r2.clear();
                }
        }
        std::cout << "Thread: " << thread_number << " buffer size: " << buffer[0].size() << std::endl;
        //Joining buffers
        //Maybe it is possible to make joining parallel?
        //It is possible to scatter buffer check between several threads, should check if it is possible to rejoin threads back then
        this->thread_buffers[thread_number] = buffer[0];
        this->thread_index_buffers[thread_number] = index_buffer[0];
        this->barrier->arrive_and_wait();
        if(thread_number == 0){
                for(int i = 1; i < this->thread_buffers.size(); i++){
                        std::cout << "AAAAAAAAAAAAAAAAAAA\n";
                        for(int j = 0; j < this->thread_buffers[i].size(); j++){
                                if(this->thread_buffers[0].size() == 0){
                                        this->thread_buffers[0].push_back(this->thread_buffers[i][j]);
                                        this->thread_index_buffers[0].push_back(this->thread_index_buffers[i][j]);
                                } else {
                                        for(int k = 0; k < this->thread_buffers[0].size(); k++){
                                                buffer_correlation.push_back(correlation(this->thread_buffers[i][j],this->thread_buffers[0][k]));
                                        }
                                        if(this->thread_buffers[0].size() < BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__ ){
                                                this->thread_buffers[0].push_back(this->thread_buffers[i][j]);
                                                this->thread_index_buffers[0].push_back(this->thread_index_buffers[i][j]);
                                        } else if (this->thread_buffers[0].size() >= BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__ ){
                                                for(int k = 0; k < this->thread_buffers[0].size(); k++){
                                                        buffer_r2.push_back(r2_score(b,this->thread_buffers[0][k]));
                                                }
                                                if(r2_score(b,this->thread_buffers[i][j]) > min<float>(buffer_r2)){
                                                        min_val_pos = find_min<float>(buffer_r2);
                                                        this->thread_buffers[0][min_val_pos.second] = this->thread_buffers[i][j];
                                                        this->thread_index_buffers[0][min_val_pos.second] = this->thread_index_buffers[i][j];
                                                }
                                        }
                                }
                        }
                        buffer_r2.clear();
                        buffer_correlation.clear();
                }
                for(int i = 0; i < amount_of_threads; i++){
                        this->thread_buffers[i] = this->thread_buffers[0];
                        this->thread_index_buffers[i] = this->thread_index_buffers[0];
                }
        }
        this->barrier->arrive_and_wait();
        buffer[0] = this->thread_buffers[thread_number];
        index_buffer[0] = this->thread_index_buffers[thread_number];
        std::cout << "Thread: " << thread_number << " buffer size after joining: " << buffer[0].size() << std::endl;
        //Additional iterations using already built models as new inputs
        //There we are not using vector parallelization for first loop but for the second one
        //Thats because buffers are usually much smaller than size of the matrix, and paralleling these loops is not as effective
        for(int iteration = 1; iteration < this->number_of_iterations; iteration++){
                A_local.col(0) = A.col(0);
                A_local.col(1) = buffer[iteration - 1][0];
                solution = A_local.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                prediction = A_local * solution;

                buffer[iteration].push_back(prediction);
                index_buffer[iteration].push_back(std::pair<int,int>(0,0));

                for(int i = 0 + thread_number; i < matrix_width; i += amount_of_threads){
                        A_local.col(0) = A.col(i);
                        for(int j = 0; j < buffer[iteration - 1].size(); j++){
                                A_local.col(1) = buffer[iteration - 1][j];
                                solution = A_local.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                                prediction = A_local * solution;

                                if (buffer[iteration].size() == 0) {
                                        buffer[iteration].push_back(prediction);
                                        index_buffer[iteration].push_back(std::pair<int,int>(i,j));
                                } else {
                                        for(int k = 0; k < buffer[iteration].size(); k++){
                                                buffer_correlation.push_back(correlation(prediction,buffer[iteration][k]));
                                        }
                                        if(buffer[iteration].size() < BUFFER_SIZE && min<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__){
                                                buffer[iteration].push_back(prediction);
                                                index_buffer[iteration].push_back(std::pair<int,int>(i,j));
                                        } else if (buffer[iteration].size() < BUFFER_SIZE && min<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__){
                                                for(int k = 0; k < buffer[iteration].size(); k++){
                                                        buffer_r2.push_back(r2_score(b,buffer[iteration][k]));
                                                }
                                                if(r2_score(b,prediction) > min<float>(buffer_r2)){
                                                        min_val_pos = find_min<float>(buffer_r2);
                                                        buffer[iteration][min_val_pos.second] = prediction;
                                                        index_buffer[iteration][min_val_pos.second] = std::pair<int,int>(i,j);
                                                }
                                        }
                                }
                        }
                        buffer_r2.clear();
                        buffer_correlation.clear();
                }

                //Now we combine built buffers into one(as we have done before with primary iteration)
                this->thread_buffers[thread_number] = buffer[iteration];
                this->thread_index_buffers[thread_number] = index_buffer[iteration];
                this->barrier->arrive_and_wait();
                if(thread_number == 0){
                        for(int i = 1; i < this->thread_buffers.size(); i++){
                                for(int j = 0; j < this->thread_buffers[i].size(); j++){
                                        if(this->thread_buffers[0].size() == 0){
                                                this->thread_buffers[0].push_back(this->thread_buffers[i][j]);
                                                this->thread_index_buffers[0].push_back(this->thread_index_buffers[i][j]);
                                        } else {
                                                for(int k = 0; k < this->thread_buffers[0].size(); k++){
                                                        buffer_correlation.push_back(correlation(this->thread_buffers[i][j],this->thread_buffers[0][k]));
                                                }
                                                if(this->thread_buffers[0].size() < BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__ ){
                                                        this->thread_buffers[0].push_back(this->thread_buffers[i][j]);
                                                        this->thread_index_buffers[0].push_back(this->thread_index_buffers[i][j]);
                                                } else if (this->thread_buffers[0].size() >= BUFFER_SIZE && max<float>(buffer_correlation) - CORR_BORDER < __DBL_EPSILON__ ){
                                                        for(int k = 0; k < this->thread_buffers[0].size(); k++){
                                                                buffer_r2.push_back(r2_score(b,this->thread_buffers[0][k]));
                                                        }
                                                        if(r2_score(b,this->thread_buffers[i][j]) > min<float>(buffer_r2)){
                                                                min_val_pos = find_min<float>(buffer_r2);
                                                                this->thread_buffers[0][min_val_pos.second] = this->thread_buffers[i][j];
                                                                this->thread_index_buffers[0][min_val_pos.second] = this->thread_index_buffers[i][j];
                                                        }
                                                }
                                        }
                                }
                                buffer_r2.clear();
                                buffer_correlation.clear();
                        }
                        for(int i = 0; i < amount_of_threads; i++){
                                this->thread_buffers[i] = this->thread_buffers[0];
                                this->thread_index_buffers[i] = this->thread_index_buffers[0];
                        }
                }
                this->barrier->arrive_and_wait();
                buffer[iteration] = this->thread_buffers[thread_number];
                index_buffer[iteration] = this->thread_index_buffers[thread_number];

        }
        if(thread_number == 0){
                this->result_buffer = buffer;
                this->result_index_buffer = index_buffer;
        }
}


//After GMDH model was build, we are ready to build our predictions
std::vector<Eigen::VectorXf> GMDHSolver::predict(Eigen::MatrixXf X){
        int ind_pred;
        std::vector<std::vector<std::pair<int,int>>> index(this->result_index_buffer[this->result_index_buffer.size() - 1].size());
        std::vector<Eigen::VectorXf> result;
        Eigen::MatrixXf A_local(matrix_height,2);
        Eigen::MatrixXf A_pred(matrix_height,2);
        Eigen::VectorXf solution;
        Eigen::VectorXf prediction;


        for(int i = 0; i < (this->result_index_buffer[this->result_index_buffer.size() - 1].size()); i++){
                std::cout << "A\n";
                ind_pred = i;
                //index.push_back(std::vector<std::pair<int,int>>());
                std::cout << "B\n";
                for(int k = number_of_iterations - 1; k >= 0; k--){
                        //std::cout << "k = " << k << std::endl; 
                        index[i].push_back(this->result_index_buffer[k][ind_pred]);
                        std::cout <<"index(" << this->result_index_buffer[k][ind_pred].first << "," << this->result_index_buffer[k][ind_pred].second << ")\n";
                        ind_pred = this->result_index_buffer[k][ind_pred].second;
                }
                std::cout << "C\n";
                std::cout << "index[i].size() = " << index[i].size() << std::endl;

                std::cout << "index[i][index[i].size() - 1].first = " << index[i][index[i].size() - 1].first << std::endl;
                std::cout << "index[i][index[i].size() - 1].second = " << index[i][index[i].size() - 1].second << std::endl ;
                fflush(stdout);
                A_local.col(0) = this->A.col(index[i].back().first);
                std::cout << "C1\n";
                fflush(stdout);
                A_local.col(1) = this->A.col(index[i].back().second);
                std::cout << "C2\n";
                solution = A_local.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                prediction = A_local * solution;
                std::cout << "D\n";
                for(int k = 1; k < number_of_iterations; k++){
                        A_local.col(0) = this->A.col(index[i][number_of_iterations - 1 - k].first);
                        std::cout << "K\n";
                        std::cout << "Size : " << this->result_buffer[k-1][index[i][number_of_iterations - 1 - k].second].rows() << " " << this->result_buffer[k-1][index[i][number_of_iterations - 1 - k].second].cols() << "\n";
                        A_local.col(1) = this->result_buffer[k-1][index[i][number_of_iterations - 1 - k].second];
                        std::cout << "Z\n";
                        solution = A_local.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                        std::cout << "L = " << index[i][number_of_iterations - 1 - k].first << "\n";
                        A_pred.col(0) = X.col(index[i][number_of_iterations - 1 - k].first);
                        std::cout << "G\n";
                        A_pred.col(1) = prediction;
                        std::cout << "M\n";
                        prediction = A_pred * solution;
                        std::cout << "E\n";
                }
                result.push_back(prediction);
                std::cout << "F\n";
        }
        return result;

}

int main(int argc, char** argv)
{
        int matrix_width, matrix_height;
        int amount_of_threads;
        if(argc != 4){
                std::cout << "Usage:" << argv[0] << " matrix_height matrix_width amount_of_threads" << std::endl;
                return -1; 
        }
        if(!sscanf(argv[1],"%d",&matrix_height)||!sscanf(argv[2],"%d",&matrix_width)||!sscanf(argv[3],"%d",&amount_of_threads)){
                std::cout << "Usage: " << argv[0] << " matrix_height matrix_width" << std::endl;
                return -1;
        }
        clock_t start = clock();
        GMDHSolver solver(matrix_height,matrix_width,BUFFER_SIZE,2,amount_of_threads);
        clock_t end = clock();
        std::cout << "Execution time: " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;
        for(int i = 0; i < solver.result_index_buffer.size(); i++){
                for(int j = 0; j < solver.result_index_buffer[i].size(); j++){
                        std::cout << "(" << solver.result_index_buffer[i][j].first << "," << solver.result_index_buffer[i][j].second << ")\n";
                }
                std::cout << std::endl;
        }
        // std::cout << "Buffer:\n";
        // for(int i = 0; i < solver.result_buffer.size(); i++){
        //         for(int j = 0; j < solver.result_buffer[i].size(); j++){
        //                 std::cout << solver.result_buffer[i][j] << std::endl;
        //         }
        //         std::cout << "\n";
        // }
        // std::cout << "\n";
        Eigen::MatrixXf sample = Eigen::MatrixXf::Random(matrix_height,matrix_width);
        std::vector<Eigen::VectorXf> predictions = solver.predict(sample);
        std::cout << "Expected b:\n" << solver.b << std::endl;
        std::cout << std::endl;
        std::cout << "Got predictions:\n";
        for(int i = 0; i < predictions.size(); i++){
                std::cout << predictions[i] << std::endl << std::endl;
                std::cout << std::endl;
        }
        return 0;
}