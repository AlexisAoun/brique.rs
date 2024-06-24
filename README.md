# brique.rs
## What is brique.rs ? 

Brique.rs is a toy project with the objective of implementing a MLP in rust without the use of any lib outside of the standard lib of rust, except (hear me out) for the rand lib. To my surprise there is not a reliable way to generate random numbers in the standard lib of rust, and having truly random numbers can make or brake in some cases the algorithm (random batch generation, random weight initialization etc.) 

A part from that everything is DIY, from the linear algebra calculations to the models themselves. I even got a DIY csv parser in there for end to end model testing.

## Current state of the project

The project is still very much a WIP but some foundations are present. Simple models can be trained but everything is slow and and the API is not user friendly. This is the objective of version release 0.2. 

## Final goal 

I want to be able to train a model that can reliably predict handwritten digits using the classic MNIST dataset, and having a little web app / web server to interact with the model.

