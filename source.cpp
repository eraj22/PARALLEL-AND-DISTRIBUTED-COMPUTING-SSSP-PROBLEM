#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <limits>
#include <set>
#include <sstream>
#include <algorithm>
#include <omp.h>
#include <metis.h>

using namespace std;

const int INF = numeric_limits<int>::max();
const int MIN_WEIGHT = 1;  // Minimum edge weight to prevent METIS errors

// Global variable for max node ID
int num_nodes = 0;

struct Edge {
    int to;
    int weight;
};

typedef unordered_map<int, vector<Edge>> Graph;

void safeInsertEdge(Graph& graph, int u, int v, int w) {
    #pragma omp critical
    {
        if (w <= 0) {
            cerr << "Correcting invalid weight " << w << " to " << MIN_WEIGHT << " for edge " << u << "-" << v << endl;
            w = MIN_WEIGHT;
        }
        if (graph.find(u) == graph.end()) {
            graph[u] = vector<Edge>();
        }
        graph[u].push_back({v, w});
    }
}

void safeDeleteEdge(Graph& graph, int u, int v, int w) {
    #pragma omp critical
    {
        if (graph.find(u) != graph.end()) {
            auto& edges = graph[u];
            edges.erase(remove_if(edges.begin(), edges.end(),
                [&](Edge e) { return e.to == v && e.weight == w; }),
                edges.end());
        }
    }
}

unordered_map<int, int> parallelDijkstra(const Graph& graph, int src) {
    unordered_map<int, int> dist;
    vector<int> nodes;
    
    // Debug output: Check if source exists
    if (graph.find(src) == graph.end()) {
        cerr << "Warning: Source node " << src << " not found in local graph!" << endl;
        // Create an empty entry for source to avoid crashes
        dist[src] = 0;
    }
    
    for (const auto& pair : graph) {
        nodes.push_back(pair.first);
        dist[pair.first] = INF;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < nodes.size(); i++) {
        dist[nodes[i]] = INF;
    }
    dist[src] = 0;

    auto cmp = [](pair<int, int> left, pair<int, int> right) { 
        return left.first > right.first; 
    };
    priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq(cmp);
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        auto it = graph.find(u);
        if (it == graph.end()) continue;

        const auto& neighbors = it->second;
        #pragma omp parallel for
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int v = neighbors[i].to;
            int w = neighbors[i].weight;
            
            // Ensure destination node exists in distance map
            if (dist.find(v) == dist.end()) {
                #pragma omp critical
                {
                    if (dist.find(v) == dist.end()) {
                        dist[v] = INF;
                    }
                }
            }
            
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                #pragma omp critical
                {
                    if (dist[u] != INF && dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                        pq.push({dist[v], v});
                    }
                }
            }
        }
    }
    return dist;
}

void readGraph(Graph& graph, const string& filename) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error: Could not open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int u, v, w;
    int edges_count = 0;
    
    // Try to read with weight
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        if (iss >> u >> v >> w) {
            safeInsertEdge(graph, u, v, w);
            safeInsertEdge(graph, v, u, w); // Undirected graph
            edges_count += 2;
        }
        else {
            // If weight not provided, use default weight of 1
            iss.clear();
            iss.str(line);
            if (iss >> u >> v) {
                safeInsertEdge(graph, u, v, 1);
                safeInsertEdge(graph, v, u, 1); // Undirected graph
                edges_count += 2;
            }
        }
    }
    
    cout << "Read " << graph.size() << " nodes and " << edges_count << " directed edges from " << filename << endl;
}

void applyChanges(Graph& graph, const string& filename, set<int>& affected) {
    ifstream changes(filename);
    if (!changes) {
        cerr << "Error: Could not open " << filename << endl;
        return;
    }

    int u, v, w;
    char op;
    int change_count = 0;
    
    while (changes >> u >> v >> w >> op) {
        if (op == 'I') {
            safeInsertEdge(graph, u, v, w);
            safeInsertEdge(graph, v, u, w); // Undirected graph
            #pragma omp critical
            affected.insert(u);
            affected.insert(v);
            change_count++;
        }
        else if (op == 'D') {
            safeDeleteEdge(graph, u, v, w);
            safeDeleteEdge(graph, v, u, w); // Undirected graph
            #pragma omp critical
            affected.insert(u);
            affected.insert(v);
            change_count++;
        }
    }
    
    cout << "Applied " << change_count << " changes, affecting " << affected.size() << " nodes" << endl;
}

// Modified: Added max_nodes parameter to function
vector<int> safePartition(const Graph& graph, int num_partitions, int max_nodes) {
    if (num_partitions <= 1) return vector<int>(graph.size(), 0);

    // Create node index mapping
    vector<int> nodes;
    unordered_map<int, int> node_to_idx;
    int idx = 0;
    for (const auto& pair : graph) {
        nodes.push_back(pair.first);
        node_to_idx[pair.first] = idx++;
    }

    // Build adjacency lists with guaranteed positive weights
    vector<idx_t> xadj(nodes.size() + 1);
    vector<idx_t> adjncy;
    vector<idx_t> adjwgt;
    
    xadj[0] = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        int u = nodes[i];
        const auto& edges = graph.at(u);
        for (const Edge& e : edges) {
            auto it = node_to_idx.find(e.to);
            if (it != node_to_idx.end()) {
                adjncy.push_back(it->second);
                adjwgt.push_back(max(MIN_WEIGHT, e.weight)); // Ensure positive weights
            }
        }
        xadj[i+1] = adjncy.size();
    }

    // METIS partitioning with error handling
    idx_t n = nodes.size();
    idx_t ncon = 1;
    idx_t nparts = num_partitions;
    vector<idx_t> part(n);
    idx_t objval;
    
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_CONTIG] = 0;  // Allow non-contiguous partitions
    options[METIS_OPTION_MINCONN] = 1;
    options[METIS_OPTION_UFACTOR] = 10;

    int ret = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(),
                                NULL, NULL, adjwgt.data(),
                                &nparts, NULL, NULL, options,
                                &objval, part.data());

    if (ret != METIS_OK) {
        cerr << "METIS fallback: Using simple block partitioning" << endl;
        for (size_t i = 0; i < nodes.size(); ++i) {
            part[i] = i % num_partitions;
        }
    }

    // Map back to original node IDs
    // Fixed: Use max_nodes instead of undefined num_nodes
    vector<int> final_part(max(*max_element(nodes.begin(), nodes.end()), max_nodes-1) + 1, 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        final_part[nodes[i]] = part[i];
    }

    return final_part;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        cout << "Running with " << world_size << " MPI processes" << endl;
    }

    double start_time = MPI_Wtime();
    
    Graph graph;
    set<int> affected;
    int source = 0;  // Default source node

    // Read graph on rank 0 only
    if (rank == 0) {
        string graph_file = "facebook_combined.txt";
        string changes_file = "facebook_changes.txt";
        
        // Check if files were provided as command-line arguments
        if (argc > 1) graph_file = argv[1];
        if (argc > 2) changes_file = argv[2];
        if (argc > 3) source = atoi(argv[3]);
        
        cout << "Reading graph from " << graph_file << endl;
        readGraph(graph, graph_file);
        
        // Apply changes if the file exists
        ifstream test_changes(changes_file);
        if (test_changes) {
            cout << "Applying changes from " << changes_file << endl;
            applyChanges(graph, changes_file, affected);
        } else {
            cout << "Changes file " << changes_file << " not found. Skipping changes." << endl;
        }
        
        cout << "Graph loaded with " << graph.size() << " nodes" << endl;
        
        // Choose a valid source node if the provided one doesn't exist
        if (graph.find(source) == graph.end() && !graph.empty()) {
            int old_source = source;
            source = graph.begin()->first;
            cout << "Source node " << old_source << " doesn't exist in the graph. Using node " << source << " instead." << endl;
        }
    }
    
    // Broadcast source node to all processes
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine max node ID
    int max_node_id = -1;
    if (rank == 0) {
        for (const auto& [node_id, _] : graph) {
            max_node_id = max(max_node_id, node_id);
            for (const Edge& e : graph[node_id]) {
                max_node_id = max(max_node_id, e.to);
            }
        }
    }
    
    // Broadcast max node ID to all processes
    MPI_Bcast(&max_node_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    num_nodes = max_node_id + 1;

    if (rank == 0) {
        cout << "Max node ID: " << max_node_id << ", Total nodes: " << num_nodes << endl;
    }

    // Partition nodes
    vector<int> partition(num_nodes, 0);
    if (rank == 0) {
        if (world_size > 1) {
            // Modified: Pass num_nodes to safePartition
            partition = safePartition(graph, world_size, num_nodes);
            
            // Count nodes per partition for debugging
            vector<int> partition_counts(world_size, 0);
            for (int i = 0; i < num_nodes; i++) {
                if (partition[i] >= 0 && partition[i] < world_size) {
                    partition_counts[partition[i]]++;
                }
            }
            
            cout << "Partition node counts: ";
            for (int i = 0; i < world_size; i++) {
                cout << "P" << i << "=" << partition_counts[i] << " ";
            }
            cout << endl;
        } else {
            // If only one process, assign all nodes to rank 0
            for (int i = 0; i < num_nodes; i++) {
                partition[i] = 0;
            }
        }
    }
    
    // Broadcast partition information to all processes
    MPI_Bcast(partition.data(), num_nodes, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process identifies its assigned nodes
    vector<int> my_nodes;
    for (int i = 0; i < num_nodes; i++) {
        if (partition[i] == rank) {
            my_nodes.push_back(i);
        }
    }

    if (rank == 0) {
        cout << "Partition complete. Starting graph distribution..." << endl;
    }

    // SIMPLIFIED AND IMPROVED GRAPH DISTRIBUTION
    // Each process needs the full graph for accurate shortest path computation
    Graph local_graph;
    
    if (rank == 0) {
        // Prepare to send the entire graph to all processes
        vector<int> all_nodes;
        vector<int> all_edges_flat;
        
        // First collect all nodes and edges in a flat format for efficient broadcasting
        for (const auto& [node_id, edges] : graph) {
            all_nodes.push_back(node_id);
            all_nodes.push_back(edges.size());  // Number of edges for this node
            
            for (const Edge& edge : edges) {
                all_edges_flat.push_back(edge.to);
                all_edges_flat.push_back(edge.weight);
            }
        }
        
        // Send number of nodes and total flat array size
        int node_count = graph.size();
        int edge_array_size = all_edges_flat.size();
        
        MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&edge_array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Send node and edge information
        MPI_Bcast(all_nodes.data(), all_nodes.size(), MPI_INT, 0, MPI_COMM_WORLD);
        if (!all_edges_flat.empty()) {
            MPI_Bcast(all_edges_flat.data(), all_edges_flat.size(), MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Also build the local_graph for rank 0
        local_graph = graph;  // Rank 0 just uses the original graph
    } else {
        // Receive the entire graph
        int node_count, edge_array_size;
        MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&edge_array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<int> all_nodes(node_count * 2);  // node_id and edge_count for each node
        MPI_Bcast(all_nodes.data(), all_nodes.size(), MPI_INT, 0, MPI_COMM_WORLD);
        
        vector<int> all_edges_flat;
        if (edge_array_size > 0) {
            all_edges_flat.resize(edge_array_size);
            MPI_Bcast(all_edges_flat.data(), edge_array_size, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // Reconstruct the graph
        int edge_idx = 0;
        for (int i = 0; i < node_count; i++) {
            int node_id = all_nodes[i*2];
            int edge_count = all_nodes[i*2+1];
            
            vector<Edge> edges;
            for (int j = 0; j < edge_count; j++) {
                Edge e;
                e.to = all_edges_flat[edge_idx++];
                e.weight = all_edges_flat[edge_idx++];
                edges.push_back(e);
            }
            local_graph[node_id] = edges;
        }
    }
    
    // Verify local graph on each process
    int local_edge_count = 0;
    for (const auto& [node, edges] : local_graph) {
        local_edge_count += edges.size();
    }
    
    cout << "Rank " << rank << " has " << local_graph.size() << " nodes and " 
         << local_edge_count << " edges. Source node " << source 
         << (local_graph.find(source) != local_graph.end() ? " exists" : " doesn't exist") 
         << " in local graph." << endl;
    
    // Process affected nodes (nodes that have changes)
    vector<int> affected_nodes;
    int affected_count = 0;
    
    if (rank == 0) {
        affected_nodes.assign(affected.begin(), affected.end());
        affected_count = affected_nodes.size();
    }
    
    MPI_Bcast(&affected_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        affected_nodes.resize(affected_count);
    }
    
    MPI_Bcast(affected_nodes.data(), affected_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local range of affected nodes for this process
    int local_start = (rank * affected_count) / world_size;
    int local_end = ((rank + 1) * affected_count) / world_size;
    
    // Process local affected nodes
    unordered_map<int, unordered_map<int, int>> local_results;
    for (int i = local_start; i < local_end; ++i) {
        if (i < affected_count) {  // Safety check
            int src = affected_nodes[i];
            cout << "Rank " << rank << " computing shortest paths from node " << src << endl;
            local_results[src] = parallelDijkstra(local_graph, src);
        }
    }
    
    // Each process computes shortest paths from the main source node
    unordered_map<int, int> source_results = parallelDijkstra(local_graph, source);
    
    // Output results from rank 0
    if (rank == 0) {
        cout << "\nFinal shortest distances after updates (from source = " << source << "):\n";
        
        // Count reachable vs unreachable nodes
        int reachable = 0, unreachable = 0;
        for (const auto& [node, dist] : source_results) {
            if (dist != INF) {
                reachable++;
            } else {
                unreachable++;
            }
            
            // Print first 20 nodes
            if (node < 20) {
                cout << "Node " << node << ": " << (dist == INF ? -1 : dist) << "\n";
            }
        }
        
        cout << "\nSummary: " << reachable << " nodes are reachable from source " << source 
             << " and " << unreachable << " nodes are unreachable (distance = -1)." << endl;
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        cout << "Total execution time: " << (end_time - start_time) << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
