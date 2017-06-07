# Neural-Network-Video-Compression
A scratch-built neural network training and application suite. I implemented a customizable deep-neural network
to train against randomized vector data to begin to experiment with video compression, wherein input seed vectors
are smaller than the output vectors. I built a Genetic Algorithm as the primary training algorithm but also inluded
experimental implementations of Particle Swarm Optimization and Ensemble Learning. The Genetic Algorithm is
multi-threaded and I also attempted to harness GPU matrix evaluation using Aparapi but never completed it.
The Genetic Algorithm manipulates the weights as an analog to DNA. At each round the DNA is evaluated by
applying its properties to a network and running that network against the training data. Experiments can be run
by changing which enum constants are used in Driver.java and by changing parameters in the training algorithms.
Feel free to experiment and explore!
