#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include <cassert>


// Use thrust to sort feature
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 32


using namespace std;

struct ForestConfig {    
    int min_children;
    int depth;
    int max_feature;
    int tree_cnt;
    int max_pos;
    float bootstrap;
    float step ;      
    int nthread;
    
    ForestConfig() {
        min_children = 10;
        depth = 8;
        tree_cnt = 1;
        max_feature = -1;        
        max_pos = -1;
        bootstrap = 0;
        step = 0.1;            
        nthread = 1;
    }
};


struct DFeature {
    vector<float> f;
    float y;    
};


struct TNode
{    
    float value;
    float splitval;
    int ind;
    int ch[2];
    float sum_y;
    float sum_sqr_y;
};


struct calculate_err {
    const float cnt;
    const float sum_y;
    const float ss_y;

    calculate_err(float _cnt, float _sum_y, float _ss_y) : cnt(_cnt), sum_y(_sum_y), ss_y(_ss_y) {}

    __host__ __device__
        float operator()(const float& sum0, const float& cnt0) const { 
            return ss_y - sum0 * sum0 / cnt0 - (sum_y-sum0) * (sum_y-sum0) / (cnt-cnt0);
        }
};

struct is_target{
    const int t;
    is_target(int _t) : t(_t) {}

    __host__ __device__
        bool operator()(const int x) {
            return x == t;
        }
};


__global__ void mapping_float(int* A, float* B, float* C, int N){

    int threadId = threadIdx.x + blockIdx.x*blockDim.x;
    if(threadId < N){
        int index = A[threadId];
        C[threadId] = B[index];
    }

}

__global__ void mapping_int(int* A, int* B, int* C, int N){

    int threadId = threadIdx.x + blockIdx.x*blockDim.x;
    if(threadId < N){
        int index = A[threadId];
        C[threadId] = B[index];
    }

}


const float EPSI = 1e-4;

unsigned long long now_rand = 1;

double get_time() {    
    struct timeval   tp;
    struct timezone  tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );	
}


void set_rand_seed(unsigned long long seed)
{
    now_rand = seed;
}


unsigned long long get_rand()
{
    now_rand = ((now_rand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
    return now_rand;
}


inline float sqr(const float &x) {
    return x * x;
}


inline int sign(const float &val) {
    if (val > EPSI) return 1;
    else if (val < -EPSI) return -1;
    else return 0;
}


double tot_parallel_time = 0.0;


class DecisionTree {
    private:    
    struct QNode {
        int nid;                
        //int left, right;
        int cnt;
        float err;
        
        QNode() {
            nid = cnt = 0;
        }
        
        QNode(const int &nid_, const int &cnt_) {
            nid = nid_;
            cnt = cnt_;
            err = 0.0f;
        }
        
        QNode(const int &nid_, const int &cnt_, const float &err_) {
            nid = nid_;
            cnt = cnt_;
            err = err_;
        }
    };    
    
    struct SplitInfo {
        int bind;
        float bsplit;
        int cnt[2];
        float sum_y[2], sum_sqr_y[2];
        float err;
        
        void update(const SplitInfo &sinfo) {
            bind = sinfo.bind;
            bsplit = sinfo.bsplit;
            cnt[0] = sinfo.cnt[0]; cnt[1] = sinfo.cnt[1];
            sum_y[0] = sinfo.sum_y[0]; sum_y[1] = sinfo.sum_y[1];
            sum_sqr_y[0] = sinfo.sum_sqr_y[0]; sum_sqr_y[1] = sinfo.sum_sqr_y[1];
            err = sinfo.err;
        }
    };
    
    struct ThreadInfo {
        int cnt0;
        float sum0, ss0;
        float last_val;
        SplitInfo spinfo;
    };
    
    vector<QNode> q;

    vector<SplitInfo> split_infos;
    // vector<int> counts;
    thrust::device_vector<int> counts;
    thrust::device_vector<float> y_list;
    thrust::device_vector<float> sqr_y_list;
    thrust::device_vector<int> positions;
    
    float *h_y_list, *h_sqr_y_list;
    int *h_positions;
    
    vector<DFeature> *features_ptr;
    
    public:
    vector<TNode> tree;        
    
    int min_children;
    int max_depth;
    
    int n; // number of instances
    int m; // number of features
    int nthread; // number of threads    
    
    
    
    private:
    void init_data() {
        q.reserve(256);
        q.resize(0);
        split_infos.reserve(256);
        split_infos.resize(0);
        tree.reserve(256);
        tree.resize(0);
    }       
    
    void update_queue() {
        vector<DFeature> &features = *features_ptr;
        vector<QNode> new_q;
        TNode new_node;
        vector< pair<int, int> > children_q_pos(q.size());
        for (int i = 0; i < q.size(); i++) {
            if (split_infos[i].bind >= 0) {
                int ii = q[i].nid;
                tree[ii].ind = split_infos[i].bind;                        
                tree[ii].splitval = split_infos[i].bsplit;                
                tree[ii].ch[0] = tree.size();
                tree[ii].ch[1] = tree.size() + 1;
                children_q_pos[i].first = new_q.size();
                children_q_pos[i].second = new_q.size() + 1;
                
                for (int c = 0; c < 2; c++) {
                    new_node.ind = -1;
                    new_node.value = split_infos[i].sum_y[c] / split_infos[i].cnt[c];
                    new_node.sum_y = split_infos[i].sum_y[c];
                    new_node.sum_sqr_y = split_infos[i].sum_sqr_y[c];
                    float err = new_node.sum_sqr_y - new_node.sum_y*new_node.sum_y/split_infos[i].cnt[c];
                    new_q.push_back(QNode(tree.size(), split_infos[i].cnt[c], err));
                    tree.push_back(new_node);                    
                }                
            }
        }
                
        for (int i = 0; i < n; i++) {
            int &pos = h_positions[i];
            if (pos >= 0 && split_infos[pos].bind >= 0) {
                if (features[i].f[split_infos[pos].bind] <=  split_infos[pos].bsplit) {
                    pos = children_q_pos[pos].first;
                } else {
                    pos = children_q_pos[pos].second;
                }                
            } else pos = -1;
        }
        
        q = new_q;
    }
       
    // set initial value and sort the column feature list
    void initial_column_feature_list(
            vector<thrust::device_vector<float>> &col_feature, 
            vector<thrust::device_vector<int>> &col_index,
            vector<int> &id_list)
    {
        
        vector<DFeature> &features = *features_ptr; 
        
        col_feature.resize(m);
        col_index.resize(m);
                
        int N = id_list.size();

        for (int i = 0; i < m; i++) {
            col_feature[i].resize(N);
            col_index[i].resize(N);
        }
        
        double start_time = get_time();

        vector<float> tmp_fea;
        for(int i = 0; i < m; i++){
            thrust::copy(id_list.begin(), id_list.end(), col_index[i].begin());
            tmp_fea.clear();
            for (int j = 0; j < N; j++) {
                int ins_id = id_list[j];
                tmp_fea.push_back(features[ins_id].f[i]);
            }
            thrust::copy(tmp_fea.begin(), tmp_fea.end(), col_feature[i].begin());
        }
        
        // sort key-value pairs
        for (int i = 0; i < m; i++) {
            thrust::sort_by_key(col_feature[i].begin(), col_feature[i].end(), col_index[i].begin());
        }        

        printf("%.3f seconds for column feature initialization\n", get_time() - start_time);
    }       
    
    // Try to parallel this function in cuda
    void find_split( int fid,
            thrust::device_vector<float> &fea_list, 
            thrust::device_vector<int> &index_list,
            vector<ThreadInfo> &tinfo_list)
    {
        int N = index_list.size();
        int n_q = tinfo_list.size();

        for (int i = 0; i < n_q; i++) {
            tinfo_list[i].cnt0 = 0;        
            tinfo_list[i].sum0 = 0.0f;
            tinfo_list[i].ss0 = 0.0f;
        }

        vector<thrust::device_vector<float>> nodes_fea;
        vector<thrust::device_vector<float>> nodes_sum0;
        vector<thrust::device_vector<float>> nodes_ss0;

        thrust::device_vector<float> ordered_y(N);
        thrust::device_vector<float> ordered_sqr_y(N);
        thrust::device_vector<int> ordered_pos(N);

        vector<thrust::device_vector<float>> nodes_err;

        nodes_fea.resize(n_q);
        nodes_sum0.resize(n_q);
        nodes_ss0.resize(n_q);
        nodes_err.resize(n_q);

        double start_time = get_time();
        int grid_size = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
        mapping_float<<<grid_size, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(index_list.data()),
                thrust::raw_pointer_cast(y_list.data()),
                thrust::raw_pointer_cast(ordered_y.data()), N);

        mapping_float<<<grid_size, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(index_list.data()),
                thrust::raw_pointer_cast(sqr_y_list.data()),
                thrust::raw_pointer_cast(ordered_sqr_y.data()), N);

        mapping_int<<<grid_size, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(index_list.data()),
                thrust::raw_pointer_cast(positions.data()),
                thrust::raw_pointer_cast(ordered_pos.data()), N);

        for (int i = 0; i < n_q; i++){
            int cnt = q[i].cnt;

            nodes_fea[i].resize(cnt);
            nodes_sum0[i].resize(cnt);
            nodes_ss0[i].resize(cnt);

            thrust::copy_if(thrust::device, fea_list.begin(), fea_list.end(), ordered_pos.begin(), nodes_fea[i].begin(), is_target(i));
            thrust::copy_if(thrust::device, ordered_y.begin(), ordered_y.end(), ordered_pos.begin(), nodes_sum0[i].begin(), is_target(i));
            thrust::copy_if(thrust::device, ordered_sqr_y.begin(), ordered_sqr_y.end(), ordered_pos.begin(), nodes_ss0[i].begin(), is_target(i));
        }
        printf("%.3f seconds for finding splits \n", get_time() - start_time);

        start_time = get_time();
        // use thrust scan to get the sum0 and ss0
        for(int i = 0; i < n_q; i++){
            int cnt = q[i].cnt;
            if (cnt < 20) continue; // don't have enough children

            thrust::exclusive_scan(nodes_sum0[i].begin(), nodes_sum0[i].end(), nodes_sum0[i].begin());
            thrust::exclusive_scan(nodes_ss0[i].begin(), nodes_ss0[i].end(), nodes_ss0[i].begin());

            ThreadInfo &tinfo = tinfo_list[i];
            int nid = q[i].nid;
            float sum_y = tree[nid].sum_y;
            float sum_sqr_y = tree[nid].sum_sqr_y;

            int n_splits = cnt - 2*min_children + 1;
            nodes_err[i].resize(n_splits);

            thrust::transform(thrust::device, nodes_sum0[i].begin()+min_children, nodes_sum0[i].end()-min_children+1,
                    counts.begin()+min_children, nodes_err[i].begin(), calculate_err(cnt, sum_y, sum_sqr_y));

            thrust::device_vector<float>::iterator iter = thrust::min_element(nodes_err[i].begin(), nodes_err[i].end());
            unsigned int index = iter - nodes_err[i].begin() + min_children;
            float err = *iter;

            if (sign(err - tinfo.spinfo.err) < 0) {                        
                SplitInfo &tbest = tinfo.spinfo;
                tbest.err = err;
                tbest.bind = fid;                    
                tbest.bsplit = (nodes_fea[i][index-1]+nodes_fea[i][index]) / 2;

                tbest.sum_y[0] = nodes_sum0[i][index];
                tbest.sum_y[1] = sum_y - nodes_sum0[i][index];

                tbest.sum_sqr_y[0] = nodes_ss0[i][index];
                tbest.sum_sqr_y[1] = sum_sqr_y - nodes_ss0[i][index];

                tbest.cnt[0] = index;
                tbest.cnt[1] = cnt - index;
            }
        }

        printf("%.3f seconds for finding splits \n", get_time() - start_time);

    }
        
    public:
    DecisionTree(vector<DFeature> &features, int max_depth, int max_feature, int max_pos, 
        int min_children, float bootstrap, int nthread) {
        
        // std::cout << "build my decison tree ..." << std::endl;
        this->n = features.size();        
        this->m = features.size() > 0 ? features[0].f.size() : 0;
        this->min_children = max(min_children, 1);
        this->max_depth = max_depth;
        this->nthread = nthread ? nthread : 1;
        this->features_ptr = &features;
        
        init_data();
        
        vector<int> id_list;
        id_list.reserve(n);
        float sum_y = 0.0;
        float sum_sqr_y = 0.0;

        // std::cout << "initial host y and sqr_y and pos ..." << std::endl;
        
        h_y_list = new float[n];
        h_sqr_y_list = new float[n];
        h_positions = new int[n]; 
            
        // process bootstrap
        for (int i = 0; i < n; i++) {
            if ((float)get_rand() / RAND_MAX >= bootstrap) {
                id_list.push_back(i);
                h_y_list[i] = features[i].y;
                h_sqr_y_list[i] = sqr(features[i].y);
                sum_y += h_y_list[i];
                sum_sqr_y += h_sqr_y_list[i];                
                h_positions[i] = 0;
            } else {
                h_positions[i] = -1;
            }
        }

        // std::cout << "initial device y and sqr_y and pos ..." << std::endl;
        y_list.resize(n);
        sqr_y_list.resize(n);
        positions.resize(n);
        thrust::copy(h_y_list, h_y_list+n, y_list.begin());
        thrust::copy(h_sqr_y_list, h_sqr_y_list+n, sqr_y_list.begin());

        counts.resize(n);
        thrust::sequence(counts.begin(), counts.end());
        
        // add the root node        
        TNode node;
        node.ind = -1;
        node.value = sum_y / (id_list.size() ? id_list.size() : 1);
        node.sum_y = sum_y;
        node.sum_sqr_y = sum_sqr_y;        
        tree.push_back(node);
        
        if (id_list.size() == 0) return;
        q.push_back(QNode(0, id_list.size(), sum_sqr_y-sum_y*sum_y/id_list.size()));  
        
        // set initial value and sort the column feature list
        std::cout << "initial column feature ..." << std::endl;

        vector<thrust::device_vector<float>> col_feature;
        vector<thrust::device_vector<int>> col_index;

        initial_column_feature_list(col_feature, col_index, id_list);

        std::cout << "initial column feature done ..." << std::endl;
        
        // build a dection tree         
        vector<ThreadInfo> tinfos;
        for (int dep = 0; dep < max_depth; dep++) {
            if (q.size() == 0) break;
            thrust::copy(h_positions, h_positions+n, positions.begin());
            
            int nq = q.size();
            split_infos.resize(q.size());
            
            tinfos.resize(q.size());                
            for (int j = 0; j < q.size(); j++) {                    
                tinfos[j].spinfo.bind = -1;
                tinfos[j].spinfo.err = q[j].err;
            }
            
            double start_time = get_time();

            // std::cout << "start finding splits ... " << std::endl;
            for (int fid = 0; fid < m; fid++) {         
                find_split(fid, col_feature[fid], col_index[fid], tinfos);
            }

            printf("%.3f seconds for finding split\n", get_time() - start_time);
            
            for (int i = 0; i < nq; i++) {
                SplitInfo &spinfo = split_infos[i];
                spinfo.bind = -1;
                if (tinfos[i].spinfo.bind >= 0 && (spinfo.bind < 0 || spinfo.err > tinfos[i].spinfo.err))
                    spinfo.update(tinfos[i].spinfo);
            }
            
            update_queue();                          
        }    
                   
        delete[] h_y_list;
        delete[] h_sqr_y_list;
        delete[] h_positions;
#ifdef cpp11        
        tree.shrink_to_fit();
#endif     
    }
        
    float predictTree(vector<float> &f) {
        int n = 0;
        while (tree[n].ind >= 0)
        {
            if (f[ tree[n].ind ] <= tree[n].splitval)
                n = tree[n].ch[0];
            else
                n = tree[n].ch[1];
        }
        return tree[n].value;
    }           
};

float cal_rmse(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    float rmse = 0;
    for (int i = 0; i < pred.size(); i++) {
        rmse += sqr(pred[i] - gt[i]);
    }
    rmse = sqrt(rmse / pred.size());
    return rmse;
}

float cal_auc(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    vector< pair<float, float> > tv;
    for (int i = 0; i < pred.size(); i++)
        tv.push_back(make_pair(pred[i], -gt[i]));
    sort(tv.begin(), tv.end());
    for (int i = 0; i < tv.size(); i++) 
        tv[i].second = -tv[i].second;
    int pos_cnt = 0, neg_cnt = 0;
    float cor_pair = 0;
    for (int i = 0; i < tv.size(); i++)
        if (tv[i].second > 0.5) {
            pos_cnt++;
            cor_pair += neg_cnt;
        } else {
            neg_cnt++;
        }
    return (neg_cnt > 0 && pos_cnt > 0) ? (cor_pair / pos_cnt / neg_cnt) : 0.0;
}

class BoostedForest {
    public:
    vector<DecisionTree*> trees;
    int depth, max_feature, max_pos, min_children, nthread;
    float bootstrap, step;
    vector<float> cur_vals, ori_vals;
    vector<float> steps;
    vector<DFeature> *val_features_ptr;

    BoostedForest() {
        val_features_ptr = NULL;
        step = 0.1;
        depth = 5;
        max_feature = max_pos = -1;
        min_children = 50;
    }
    
    void set_val_data(vector<DFeature> &data) {
        val_features_ptr = &data;
    }
    
    void buildForest(vector<DFeature> &features, int num_tree, int depth_, int max_feature_, 
        int max_pos_, int min_children_, float bootstrap_, float step_, int nthread_) {        
                
        depth = depth_;
        max_feature = max_feature_;
        max_pos = max_pos_;
        min_children = min_children_;
        bootstrap = bootstrap_;
        step = step_;
        nthread = nthread_;
        if (max_feature < 0) max_feature = int(sqrt(features[0].f.size()) + 1);
        
        cur_vals = vector<float>(features.size());
        ori_vals = vector<float>(features.size());
        for (int i = 0; i < features.size(); i++)
            ori_vals[i] = features[i].y;
        
        vector<float> val_vals;
        vector<float> pred_vals;
        if (val_features_ptr != NULL) {
            vector<DFeature> &val_features = *val_features_ptr;
            pred_vals = vector<float>(val_features.size());
            val_vals = vector<float>(val_features.size());
            for (int i = 0; i < val_features.size(); i++)
                val_vals[i] = val_features[i].y;
        }        
        
        float train_rmse = -1, test_rmse = -1;
        float train_auc = -1, test_auc = -1;        
                
        double train_time = 0.0;
        double start_time = get_time();
        for (int i = 0; i < num_tree; i++)
        {
            double iter_start_time = get_time();
            for (int j = 0; j < features.size(); j++)
                features[j].y = ori_vals[j] - cur_vals[j];
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);
            
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;            
            }    
            
            train_time += get_time() - iter_start_time;
            
            train_rmse = cal_rmse(cur_vals, ori_vals);
            train_auc = cal_auc(cur_vals, ori_vals);
            
            if (val_features_ptr != NULL) {                
                vector<DFeature> &val_features = *val_features_ptr;
                for (int j = 0; j < val_features.size(); j++) {
                    pred_vals[j] += dt->predictTree(val_features[j].f) * step;                    
                }
                test_rmse = cal_rmse(pred_vals, val_vals);
                test_auc = cal_auc(pred_vals, val_vals);
            }
            
            steps.push_back(step);
            
            printf("iter: %d, train_rmse: %.6lf, test_rmse: %.6lf, tree_size: %d\n", i + 1, train_rmse, test_rmse, dt->tree.size());
            printf("train_auc: %.6lf, test_auc: %.6lf\n", train_auc, test_auc);            
            printf("%.3f seconds passed, %.3f seconds in parallel,%.3f seconds in training\n", get_time() - start_time, tot_parallel_time, train_time);
        }
                
        FILE *fout = fopen("time.out", "w");
        fprintf(fout, "%.3f\n", train_time);
        fclose(fout);
        
        for (int j = 0; j < features.size(); j++)
            features[j].y = ori_vals[j];
    }

    void buildForest(vector<DFeature> &features, ForestConfig &conf) {
        buildForest(features, conf.tree_cnt, conf.depth, conf.max_feature, conf.max_pos, 
            conf.min_children, conf.bootstrap, conf.step, conf.nthread);
    }
    
    
    void addTree(vector<DFeature> &features) {        
        addTree(features, 1);
    }

    void addTree(vector<DFeature> &features, int treecnt) {        
        for (int j = 0; j < features.size(); j++) {
            ori_vals[j] = features[j].y;            
        }
        while (treecnt--) {
            for (int j = 0; j < features.size(); j++) {                
                features[j].y = ori_vals[j] - cur_vals[j];
            }
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);                          
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;
            }
            steps.push_back(step);
        }
        for (int j = 0; j < features.size(); j++) {            
            features[j].y = ori_vals[j];
        }
    }
    
    void set_step(float step_) {
        step = step_;
    }
    
    float predictForest(vector<float> &f) {
        float ret = 0;        
        for (int j = 0; j < trees.size(); j++) {
            ret += trees[j]->predictTree(f) * steps[j];            
        }        
        return ret;
    }
};


vector<DFeature> read_csv(string fname = "cleaned.csv", bool skip_first_row=false, bool skip_first_col=false){

    /* Read data set from csv file, save as vector<DFeature> 
     * Input:
     *      fname: input csv file name
     *      skip_first_row: skip the first row or not
     *      skip_first_col: skip the first column or not
     */

    int count = 0;

    vector<DFeature> data;

    string line, num;
    DFeature row;

    fstream file (fname, ios::in);
    if(file.is_open()) {
        while(getline(file, line)) {
            count++;
            // if (count % 10 != 0) continue;
            row.f.clear();

            stringstream str(line);

            while(getline(str, num, ','))
                row.f.push_back(stof(num));
            row.y = row.f.back();
            row.f.pop_back();
            data.push_back(row);
        }
    }

    return data;
}


int main(int argc, char* argv[]){
    // Load data from csv file
    vector<DFeature> data;
    data = read_csv();
    // Check input data
    std::cout << "Number of events: " << data.size() << std::endl;
    for (int i=0; i<25; i++){
        std::cout << data[0].f[i] << ", ";
    }
    std::cout << data[0].y << std::endl;

    // Forest config
    cout << "Init BDT config" << endl;
    ForestConfig my_config = ForestConfig();
    // Build GBDT in parallel
    cout << "Create a new bdt model" << endl;
    BoostedForest my_bdt = BoostedForest();

    cout << "Start building my bdt" << endl;
    my_bdt.buildForest(data, my_config);

    return 0;
}
