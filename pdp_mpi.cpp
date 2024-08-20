#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <limits.h> 
#include <chrono>
#include <cstring>

#include <mpi.h>
#include <omp.h>

#define EDGE_USED 1
#define EDGE_NOT_USED 0

#define NOT_USED -1
#define MAX_COLORS 2
#define RED 0
#define GREEN 1

enum MsgType : int
{
	NumStates = 0,
	StateBufSize,
	States,
	Result
};

struct Edge
{
	Edge() : vert1(0), vert2(0), weight(0) {}

	Edge(size_t _vert1, size_t _vert2, int _weight)
		: vert1(_vert1), vert2(_vert2), weight(_weight)
	{}

	size_t vert1;
	size_t vert2;
	int weight;
};

typedef std::vector<uint8_t> MaskBuffer;

struct State
{
	State() = default;

	State(const MaskBuffer& _buffer, int _max_weight)
	: buffer(_buffer), max_weight(_max_weight)
	{
		buffer.reserve(_buffer.capacity());
	}

	void write_buffer(u_char* dst)
	{
		*((int*)dst) = max_weight;
		u_char* buf_prt = dst + sizeof(int);
		std::memcpy(buf_prt, buffer.data(), buffer.size());
	}

	void read_buffer(u_char* src, size_t size, size_t capacity)
	{
		buffer.reserve(capacity);
		buffer.resize(size);
		max_weight = *((int*)src);
		u_char* buf_ptr = src + sizeof(int);
		std::memcpy(buffer.data(), buf_ptr, size);
	}

	int max_weight = 0;
	MaskBuffer buffer;
};

class Graph
{
public:

	~Graph()
	{
		for (size_t i = 0; i < parallel_states.size(); i++)
		{
			delete parallel_states[i];
		}
	}

	void set_depth_to_parallel(size_t _depth)
	{
		depth_to_parallel = _depth;
	}

	void init(size_t _vert_num)
	{
		vert_num = _vert_num;
		graph.reserve(vert_num * vert_num);
		for (size_t i = 0; i < vert_num; i++)
		{
			graph[d2_to_d1(vert_num, i)] = 0;
		}
	}

	void add_edge(Edge _edge)
	{
		edges.push_back(_edge);

		size_t neig_i = ++graph[d2_to_d1(vert_num, _edge.vert1)];
		graph[d2_to_d1(vert_num, _edge.vert1, neig_i)] = edges.size() - 1;
		
		neig_i = ++graph[d2_to_d1(vert_num, _edge.vert2)];
		graph[d2_to_d1(vert_num, _edge.vert2, neig_i)] = edges.size() - 1;
	}

	void find_solution()
	{
		int my_rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

		if (my_rank != 0)
		{
			slave_find_solution();
		}
		else
		{
			master_find_solution();
		}
	}

	void slave_find_solution()
	{
		int my_rank;
		int num_proc = 2;
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

		//std::string filename("process");
		//filename += std::to_string(my_rank);

		//std::ofstream output_file(filename);

		//output_file << "Slave " << my_rank << ": start algorithm" << std::endl;

		size_t state_buf_size = edges.size() - depth_to_parallel + 1;
		size_t max_num_states = ceil(pow(2, state_buf_size) / num_proc);
		size_t state_size = sizeof(int) + state_buf_size;
		size_t buffer_size = state_size * max_num_states + sizeof(size_t);
		u_char* buffer_ptr = new u_char[buffer_size];

		//output_file << "Slave " << my_rank << ": num_edges: " << edges.size() << ", edges_to_stop: " << state_buf_size << std::endl;
		//output_file << "Slave " << my_rank << ": waiting for master's msg" << std::endl;
		
		MPI_Status status;
		MPI_Recv(buffer_ptr, buffer_size, MPI_UNSIGNED_CHAR, 0, States, MPI_COMM_WORLD, &status);

		size_t num_states = *((size_t*)buffer_ptr);
		u_char* states_buffer_ptr = buffer_ptr + sizeof(size_t);
		parallel_states.reserve(num_states);

		//output_file << "Test: state_buf_size: " << state_buf_size << std::endl;

		//output_file << "Slave " << my_rank << ": msg received. num_states: " << num_states << ". reading started" << std::endl;

		for (size_t state_i = 0; state_i < num_states; state_i++)
		{
			size_t buffer_i = state_i * state_size;
			State* state = new State();
			state->read_buffer(states_buffer_ptr + buffer_i, state_buf_size, edges.size());

			/*
			output_file << '\t' << state_i << "\t weight: " << state->max_weight;
			output_file << "\tbuffer: {";
			MaskBuffer& buffer = state->buffer;
			for (size_t j = 0; j < buffer.size(); j++)
			{
				if (j != 0) { output_file << ", "; }
				output_file << (bool)buffer[j];
			}
			output_file << "}" << std::endl;
			*/

			parallel_states.push_back(state);
		}


		delete buffer_ptr;

		parallel_find_solution();

		//output_file << "Slave " << my_rank << ": result weight: " << best_state.max_weight << std::endl;

		size_t res_buffer_size = sizeof(int) + edges.size();
		u_char* res_buffer = new u_char[res_buffer_size];
		best_state.write_buffer(res_buffer);

		MPI_Send(res_buffer, res_buffer_size, MPI_UNSIGNED_CHAR, 0, Result, MPI_COMM_WORLD);

		delete res_buffer;

		//output_file << "Slave " << my_rank << ": result was sent to master" << std::endl;
		//output_file << "Slave " << my_rank << ": process finished" << std::endl;
	}

	void master_find_solution()
	{
		//std::string filename("process0");

		//std::ofstream output_file(filename);

		//output_file << "Master: start algorithm" << std::endl;

		int num_proc = 1;
		MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

		MaskBuffer used_edges;
		used_edges.reserve(edges.size());

		int graph_weight = 0;
		for (Edge& edge : edges) { graph_weight += edge.weight; }

		size_t idx_to_stop = edges.size() - depth_to_parallel;
		size_t parallel_states_capacity = static_cast<size_t>(pow(2, idx_to_stop + 1));
		parallel_states.reserve(parallel_states_capacity);
		//output_file << "Master: num_edges: " << edges.size() << ", edges_to_stop: " << idx_to_stop + 1 << std::endl;

		auto start = std::chrono::high_resolution_clock::now();

		construct_subgraph(used_edges, graph_weight);

		//output_file << "Master: algorithm phase 1 finished. start prepairing buffer" << std::endl;

		size_t max_num_states = ceil(parallel_states_capacity / (float)num_proc);
		size_t state_buf_size = idx_to_stop + 1;
		size_t state_size = sizeof(int) + state_buf_size;
		size_t buffer_size = max_num_states * state_size + sizeof(size_t);

		u_char* buffer_ptr = new u_char[buffer_size];
		u_char* states_buffer_ptr = buffer_ptr + sizeof(size_t);

		for (size_t dest_i = 1; dest_i < num_proc; dest_i++)
		{
			//output_file << "Master: start sending data to process: " << dest_i << std::endl;
			size_t j = 0;
			for (; j < max_num_states; j++)
			{
				size_t state_idx = num_proc * j + dest_i;
				if (state_idx >= parallel_states_capacity)
				{
					break;
				}
				size_t buf_idx = j * state_size;
				u_char* state_ptr = states_buffer_ptr + buf_idx;
				parallel_states[state_idx]->write_buffer(state_ptr);

				/*
				output_file << '\t' << j << "\t weight: " << *((int*)state_ptr);
				output_file << "\tbuffer: {";
				u_char* buffer = state_ptr + sizeof(int);
				for (size_t i = 0; i < state_buf_size; i++)
				{
					if (i != 0) { output_file << ", "; }
					output_file << (bool)buffer[i];
				}
				output_file << "}" << std::endl;
				*/

			}
			((size_t*)buffer_ptr)[0] = j;
			MPI_Send(buffer_ptr, buffer_size, MPI_UNSIGNED_CHAR, dest_i, States, MPI_COMM_WORLD);
			
			//output_file << "Master: data were sent to process: " << dest_i << std::endl;
		}

		delete buffer_ptr;

		//output_file << "Master: start algorithm phase 2"<< std::endl;

		parallel_find_solution(num_proc);
		
		//output_file << "Master: allgorithm was finished. result: " << best_state.max_weight << std::endl; 
		//output_file << "Master: waiting for results from other processes" << std::endl;

		size_t res_buffer_size = sizeof(int) + edges.size();
		u_char* res_buffer = new u_char[res_buffer_size];
		MPI_Status status;

		for (size_t src_i = 1; src_i < num_proc; src_i++)
		{
			MPI_Recv(res_buffer, res_buffer_size, MPI_UNSIGNED_CHAR, src_i, Result, MPI_COMM_WORLD, &status);
			
			int res_weight = *((int*)res_buffer);
			//output_file << "Master: result from process: " << src_i <<" were received. weight: " << res_weight << std::endl;
			
			if (res_weight > best_state.max_weight)
			{
				best_state.read_buffer(res_buffer, edges.size(), edges.size());
			}
		}

		delete res_buffer;

		//output_file << "Master: process finished" << std::endl;

		auto end = std::chrono::high_resolution_clock::now();

		std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	}

	void parallel_find_solution(size_t i_step = 1)
	{
		omp_init_lock(&best_weight_lock);

		#pragma omp parallel for schedule(dynamic)//schedule(static, 1)
		for (size_t i = 0; i < parallel_states.size(); i += i_step)
		{
			construct_subgraph(parallel_states[i]->buffer, parallel_states[i]->max_weight);
		}

		omp_destroy_lock(&best_weight_lock);
	}

	void construct_subgraph(MaskBuffer& _edges_mask, int _subgraph_weight)
	{
		size_t _curr_buffer_i = _edges_mask.size();
		size_t remaining_depth = _edges_mask.capacity() - _curr_buffer_i;

		if (remaining_depth < depth_to_parallel)
		{
			omp_set_lock(&best_weight_lock);
			if (_subgraph_weight <= best_state.max_weight) 
			{ 
				omp_unset_lock(&best_weight_lock);
				return; 
			} 
			omp_unset_lock(&best_weight_lock);
		}

		if (_edges_mask.size() != _edges_mask.capacity())
		{
			int edge_weight = edges[_curr_buffer_i].weight;
			_edges_mask.push_back(EDGE_NOT_USED);

			if (remaining_depth == depth_to_parallel)
			{
				parallel_states.push_back(new State(_edges_mask, _subgraph_weight - edge_weight));
				_edges_mask[_curr_buffer_i] = EDGE_USED;
				parallel_states.push_back(new State(_edges_mask, _subgraph_weight));
			}
			else
			{
				construct_subgraph(_edges_mask, _subgraph_weight - edge_weight);
				_edges_mask[_curr_buffer_i] = EDGE_USED;
				construct_subgraph(_edges_mask, _subgraph_weight);
			}

			_edges_mask.pop_back();
		}
		else
		{
			try_subgraph(_edges_mask, _subgraph_weight);
		}
	}

	void try_subgraph(const MaskBuffer& _edges_mask, int _subgraph_weight)
	{
		if (check_subgraph(_edges_mask))
		{
			omp_set_lock(&best_weight_lock);
			if (best_state.max_weight < _subgraph_weight)
			{
				best_state.max_weight = _subgraph_weight;
				best_state.buffer = _edges_mask;
			}
			omp_unset_lock(&best_weight_lock);
		}
	}

	bool check_subgraph(const MaskBuffer& _edges_mask)
	{
		std::vector<int> subgraph(vert_num, NOT_USED);
		std::queue<size_t> bfs_queue;
		
		bool edge_found = false;
		for (size_t edge_i = 0; edge_i < _edges_mask.size(); edge_i++)
		{
			if (_edges_mask[edge_i] == EDGE_NOT_USED) { continue; }
			Edge& edge = edges[edge_i];
			subgraph[edge.vert1] = subgraph[edge.vert2] = MAX_COLORS;
			if (!edge_found) 
			{ 
				edge_found = true;
				bfs_queue.push(edge.vert1); 
				subgraph[edge.vert1] = RED;
			}
		}
		
		bool success = true;
		while (bfs_queue.size() > 0)
		{
			size_t vert_i = bfs_queue.front();
			bfs_queue.pop();

			for (size_t i = 1; i <= graph[d2_to_d1(vert_num, vert_i)]; i++)
			{
				size_t edge_i = graph[d2_to_d1(vert_num, vert_i, i)];
				if (_edges_mask[edge_i] == EDGE_NOT_USED) { continue; }

				size_t next_vert_i = edges[edge_i].vert1;
				if (next_vert_i == vert_i)
				{
					next_vert_i = edges[edge_i].vert2;
				}

				if (subgraph[next_vert_i] == NOT_USED) { continue; }
				if (subgraph[next_vert_i] == MAX_COLORS)
				{
					subgraph[next_vert_i] = (subgraph[vert_i] + 1) % MAX_COLORS;
					bfs_queue.push(next_vert_i);
				}
				else if (subgraph[next_vert_i] == subgraph[vert_i])
				{
					success = false;
					break;
				}
			}
			if (!success) { break; }
		}

		for (size_t vert_i = 0; vert_i < subgraph.size() && success; vert_i++)
		{
			if (subgraph[vert_i] == MAX_COLORS)
			{
				success = false;
				break;
			}
		}

		return success;
	}

	void print_solution()
	{
		std::cout << "Max weight: " << best_state.max_weight << std::endl;

		std::cout << '{';
		size_t num_edges = get_edge_num();
		for (size_t edge_i = 0; edge_i < num_edges; edge_i++)
		{
			if (best_state.buffer[edge_i] == EDGE_USED)
			{
				Edge& edge = edges[edge_i];
				std::cout << '(' << edge.vert1 << ',' << edge.vert2 << ')';
				if (edge_i < num_edges - 1)
				{
					std::cout << ',';
				}
			}
		}
		std::cout << '}' << std::endl;
	}

private:

	size_t get_edge_num() const { return edges.size(); }

	size_t d2_to_d1(size_t _rows_num = 0, size_t _col_i = 0, size_t _row_i = 0)
	{
		return _col_i * _rows_num + _row_i;
	}

	size_t vert_num = 0;
	std::vector<Edge> edges;
	std::vector<size_t> graph;
	std::vector<State*> parallel_states;

	omp_lock_t best_weight_lock;
	size_t depth_to_parallel = 13;
	
	State best_state;
};

int read_file(const char* _filename, Graph& _graph)
{
	std::ifstream input_file;
	input_file.open(_filename);
	if (input_file.fail())
	{
		std::cout << "failed to open file: " << _filename << std::endl;
		return -3;
	}

	size_t vert_num = 0;
	input_file >> vert_num;
	_graph.init(vert_num);
	for (size_t vert1_i = 0; vert1_i < vert_num; vert1_i++)
	{
		size_t weight = 0;
		size_t vert2_i = 0;
		for (; vert2_i <= vert1_i; vert2_i++)
		{
			input_file >> weight;
		}
		for (; vert2_i < vert_num; vert2_i++)
		{
			input_file >> weight;
			if (weight == 0) { continue; }
			_graph.add_edge(Edge(vert1_i, vert2_i, weight));
		}
	}

	input_file.close();

	return 0;
}

const char* test_file = "../graf_bpo/graf_10_3.txt";

int main(int argc, char** argv)
{

	MPI_Init(&argc, &argv);

	Graph graph;
	size_t num_threads = 1;
	for (int i = 1; i < argc - 1; i += 2)
	{
		if (argv[i][0] != '-')
		{
			std::cout << "incorrect syntax: " << argv[i] << std::endl;
			return -1;
		}
		switch (argv[i][1])
		{
			case 't':
			{
				num_threads = std::atoi(argv[i + 1]);
			}
			break;
			case 'd':
			{
				graph.set_depth_to_parallel(std::atoi(argv[i + 1]));
			}
			break;
			default:
			{
				std::cout << "unknown option: -" << argv[i][1] << std::endl;
				return -2;
			}
		}
	}
	
	//int res = read_file(test_file, graph);
	int res = read_file(argv[argc - 1], graph);
	if (res < 0) { return res; }

	omp_set_num_threads(num_threads);
	
	graph.find_solution();
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0)
		graph.print_solution();

  	MPI_Finalize();
	
	return 0;
}
