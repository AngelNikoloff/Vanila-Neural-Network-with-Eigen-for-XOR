#pragma once

// Author:		Angel Nikoloff
// Date:		08-01-20
// Description:	Simple vanila c++ neural network for XOR solving made with Eigen Library - for education purpose; 

#include "E:\AMS\eigen\eigen3\Eigen\Dense" // http://eigen.tuxfamily.org/index.php?title=Main_Page

#include <vector>
#include <iostream>

class SimpleXorNet_Eigen
{
public:
	size_t epochs = 10000;
	double alfa = 1;                       // Learning rate;
	double error = 0;
	const double target_error = 0.05;
public:
	Eigen::MatrixXd input_XOR;
	Eigen::MatrixXd Target_XOR;

	std::vector<size_t> topology = { 2, 2, 1 };
	size_t input_size = topology[0];    // input vector size;
	size_t hidden_size = topology[1];   // hidden vectore size;
	size_t output_size = topology[2];   // output vectore size;

	Eigen::ArrayXd i_v = Eigen::ArrayXd::Zero(input_size);   // input vector;
	Eigen::ArrayXd h_v = Eigen::ArrayXd::Zero(hidden_size);  // Hidden vector - hidden state;
	Eigen::ArrayXd o_v = Eigen::ArrayXd::Zero(output_size);  // Output vector;
	Eigen::ArrayXd t_v = Eigen::ArrayXd::Zero(output_size);  // Target vector;

	//========================================================================== HIDDEN LAYER;
	Eigen::MatrixXd m_h_W = Eigen::MatrixXd::Random(hidden_size, input_size);    // Hidden layer weight matrix;  // row / cols;
	Eigen::ArrayXd  m_h_b = Eigen::ArrayXd::Random(hidden_size);                 // Bias hidden Weights Vectors;

	Eigen::MatrixXd d_h_W = Eigen::MatrixXd::Zero(m_h_W.rows(), m_h_W.cols());   // Delta hidden layer weights;
	Eigen::ArrayXd  d_h_b = Eigen::ArrayXd::Zero(m_h_b.size());                  // Delta hidden layer bias weights;
	//========================================================================== OUTPUT LAYER;
	Eigen::MatrixXd m_o_W = Eigen::MatrixXd::Random(output_size, hidden_size);   // Output layer weight matrix;
	Eigen::ArrayXd  m_o_b = Eigen::ArrayXd::Random(output_size);                 // Bias output Weights Vectors;

	Eigen::MatrixXd d_o_W = Eigen::MatrixXd::Zero(m_o_W.rows(), m_o_W.cols());   // Delta Output layer weights;
	Eigen::ArrayXd  d_o_b = Eigen::ArrayXd::Zero(m_o_b.size());                  // Delta Output layer bias weights;
	 //==========================================================================
	Eigen::ArrayXd derivativ = Eigen::ArrayXd::Zero(output_size);
	Eigen::ArrayXd delta_err = Eigen::ArrayXd::Zero(output_size);
	Eigen::ArrayXd gradient = Eigen::ArrayXd::Zero(output_size);
	//==========================================================================
	SimpleXorNet_Eigen()
	{
		//=============================================================== SET INPUT VECTOR VALUES;
		input_XOR.resize(4, 2);
		input_XOR <<
			1, 1,
			0, 1,
			1, 0,
			0,0;
		//=============================================================== SET OUTPUT VECTOR VALUES;
		Target_XOR.resize(4, 1);
		Target_XOR <<
			0,
			1,
			1,
			0;
		std::cout << std::endl << input_XOR << std::endl << std::endl << Target_XOR << std::endl;
		//===============================================================  SET CUSTOM WEIGHTS VALUES;

		// comment following to use random values;

		m_h_W <<
			0, 0.25,
			0.5, 1;

		m_h_b <<
		    0, 0;

		m_o_W <<
			0.25, 0.75;

		m_o_b <<
			0.5;
		//===============================================================
	}
	//=========================================================================================================================================== EIGEN VARIANT;
	void feedforward(Eigen::ArrayXd &input)
	{
		//==================================================================== calc hidden vector;
		h_v = (m_h_W * input.matrix()).array() + m_h_b;
		h_v = h_v.tanh();
		//==================================================================== calc output vector;
		o_v = (m_o_W * h_v.matrix()).array() + m_o_b;
		o_v = o_v.tanh();
		//==================================================================== calc error;
		Eigen::ArrayXd delta_err = Eigen::ArrayXd::Zero(output_size);
		delta_err = abs(t_v - o_v);
		error = error + delta_err.sum();
		//====================================================================
	}
	void backpropogate()
	{
		//====================================================================  Calc Output layer gradient and delta weights;
		derivativ = 1 - o_v*o_v;
		delta_err = t_v - o_v;
		gradient = derivativ * delta_err;
		//===============================
		d_o_W = gradient.matrix() * h_v.matrix().transpose();
		d_o_b = gradient*m_o_b;
		//==================================================================== Calc Hidden layer gradient and delta weights;
		derivativ = 1 - h_v*h_v;
		delta_err = (m_o_W.transpose() * gradient.matrix()).array();
		gradient = derivativ * delta_err;
		//===============================
		d_h_W = gradient.matrix() * i_v.matrix().transpose();
		d_h_b = gradient.matrix() * m_h_b.matrix().transpose();
		//==================================================================== Update weights;

		// update output layer weights and bias weight;
		m_o_W = m_o_W + alfa*d_o_W;
		m_o_b = m_o_b + alfa*d_o_b;

		// update hidden layer weights and bias weight;
		m_h_W = m_h_W + alfa*d_h_W;
		m_h_b = m_h_b + alfa*d_h_b;
		//====================================================================
	}
	//===========================================================================================================================================
	void run()
	{
		epochs = 10000;
		double tatget_error = 0.05; // we reach the target error of 0.05 in 1248 epoch;
		//====================
		size_t epoch = 0;
		double last_error = 0;
		while (epoch < epochs)
		{
			epoch++;
			error = 0;
			for (int task = 0; task < 4; task++) // we have 4 training examples;
			{
				i_v = input_XOR.row(task);       // set input vector values;
				t_v = Target_XOR.row(task);      // set target vector values;

				feedforward(i_v);
				backpropogate();
			}
			std::cout << "epoch: " << epoch << "  /  error: " << error << "  /  delta error change: " << error - last_error << std::endl;
			last_error = error;

			if (error < tatget_error) break;
		}
	}
} anet;





