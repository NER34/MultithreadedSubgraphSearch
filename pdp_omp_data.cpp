#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>
#include <limits.h> 
#include <chrono>

#include <omp.h>

#define EDGE_USED 1
#define EDGE_NOT_USED 2

#define NOT_USED -1
#define MAX_COLORS 2
#define RED 0
#define GREEN 1

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
	State(const MaskBuffer& _buffer, int _max_weight)
	: buffer(_buffer), max_weight(_max_weight)
	{}

	int max_weight = 0;
	MaskBuffer buffer;
};

class Graph
{
public:

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

	void find_solution(size_t num_threads)
	{
		MaskBuffer used_edges;
		used_edges.resize(edges.size(), 0);
		int graph_weight = 0;
		for (Edge& edge : edges) { graph_weight += edge.weight; }

		size_t idx_to_stop = used_edges.size() - depth_to_parallel;
		size_t capacity = static_cast<size_t>(pow(2, idx_to_stop + 1));
		parallel_states.reserve(capacity);

		auto start = std::chrono::high_resolution_clock::now();

		omp_set_num_threads(num_threads);
		omp_init_lock(&best_weight_lock);
		
		construct_subgraph(used_edges, 0, graph_weight);
		
		#pragma omp parallel for schedule(dynamic)//schedule(static, 1)
		for (size_t i = 0; i < parallel_states.size(); i++)
		{
			size_t curr_buffer_i = edges.size() - depth_to_parallel;
			construct_subgraph(parallel_states[i]->buffer, curr_buffer_i + 1, parallel_states[i]->max_weight);
			delete parallel_states[i];
		}
		
		omp_destroy_lock(&best_weight_lock);

		auto end = std::chrono::high_resolution_clock::now();
		
		std::cout << "Tume: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	}

	void construct_subgraph(MaskBuffer _edges_mask, size_t _curr_buffer_i, int _subgraph_weight)
	{
		omp_set_lock(&best_weight_lock);
		if (_subgraph_weight <= best_weight) 
		{ 
			omp_unset_lock(&best_weight_lock);
			return; 
		} 
		omp_unset_lock(&best_weight_lock);

		if (_curr_buffer_i != _edges_mask.size())
		{
			int edge_weight = edges[_curr_buffer_i].weight;

			size_t remaining_depth = _edges_mask.size() - _curr_buffer_i;
			if (remaining_depth == depth_to_parallel)
			{
				_edges_mask[_curr_buffer_i] = EDGE_NOT_USED;
				parallel_states.push_back(new State(_edges_mask, _subgraph_weight - edge_weight));
				_edges_mask[_curr_buffer_i] = EDGE_USED;
				parallel_states.push_back(new State(_edges_mask, _subgraph_weight));
			}
			else
			{
				_edges_mask[_curr_buffer_i] = EDGE_NOT_USED;
				construct_subgraph(_edges_mask, _curr_buffer_i + 1, _subgraph_weight - edge_weight);
				_edges_mask[_curr_buffer_i] = EDGE_USED;
				construct_subgraph(_edges_mask, _curr_buffer_i + 1, _subgraph_weight);
			}
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
			if (best_weight < _subgraph_weight)
			{
				best_weight = _subgraph_weight;
				best_subgraph_edges = _edges_mask;
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
		std::cout << "Max weight: " << best_weight << std::endl;

		std::cout << '{';
		size_t num_edges = get_edge_num();
		for (size_t edge_i = 0; edge_i < num_edges; edge_i++)
		{
			if (best_subgraph_edges[edge_i] == EDGE_USED)
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
	size_t depth_to_parallel = 6;
	
	int best_weight = 0;
	MaskBuffer best_subgraph_edges;
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
	
	graph.find_solution(num_threads);
	
	graph.print_solution();
	
	return 0;
}
