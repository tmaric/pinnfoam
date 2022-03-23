/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 Tomislav Maric, TU Darmstadt 
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pinnFoam

Description

\*---------------------------------------------------------------------------*/

// libtorch
#include <torch/torch.h>
#include "ATen/Functions.h"
#include "ATen/core/interned_strings.h"
#include "torch/nn/modules/activation.h"
#include "torch/optim/lbfgs.h"
#include "torch/optim/rmsprop.h"

// STL 
#include <algorithm>
#include <random> 
#include <numeric>
#include <cmath>
#include <filesystem>

// OpenFOAM 
#include "fvCFD.H"

// libtorch-OpenFOAM data transfer
#include "torchFunctions.C"
#include "fileNameGenerator.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "volFieldName",
        "string",
        "Name of the volume (cell-centered) field approximated by the neural network."
    );

    argList::addOption
    (
        "approximator",
        "string",
        "Type name of the MLP field approximator."
    );

    argList::addOption
    (
        "hiddenLayers",
        "int,int,int,...",
        "A sequence of hidden-layer depths."
    );

    argList::addOption
    (
        "optimizerStep",
        "double",
        "Step of the optimizer."
    );

    argList::addOption
    (
        "epsilon",
        "<double>",
        "Training error tolerance."
    );

    argList::addOption
    (
        "maxIterations",
        "<int>",
        "Max number of iterations."
    );
    
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // Initialize hyperparameters 
    
    // - NN architecture 
    DynamicList<label> hiddenLayers;
    scalar optimizerStep;
    // - Training error tolerance.
    scalar epsilon;
    // - Maximal number of training iterations.
    std::size_t maxIterations;
    
    // - Initialize hyperparameters from command line arguments if they are provided
    if (args.found("hiddenLayers") && 
        args.found("optimizerStep") &&
        args.found("epsilon") && 
        args.found("maxIterations"))
    {
        hiddenLayers = args.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = args.get<scalar>("optimizerStep");
        epsilon = args.get<scalar>("epsilon");
        maxIterations = args.get<label>("maxIterations");
    } 
    else // Initialize from system/fvSolution.AI.approximator sub-dict.
    {
        const fvSolution& fvSolutionDict (mesh);
        const dictionary& aiDict = fvSolutionDict.subDict("AI");

        hiddenLayers = aiDict.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = aiDict.get<scalar>("optimizerStep");
        epsilon = aiDict.get<scalar>("epsilon");
        maxIterations = aiDict.get<label>("maxIterations");
    }
    
    // Use double-precision floating-point arithmetic. 
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::kDouble)
    );
    
    // Construct the MLP 
    torch::nn::Sequential nn;
    // - Input layer are always the 3 spatial coordinates in OpenFOAM, 2D
    //   simulations are pseudo-2D (single cell-layer).
    nn->push_back(torch::nn::Linear(3, hiddenLayers[0])); 
    nn->push_back(torch::nn::GELU()); // FIXME: RTS activation function.
    // - Hidden layers
    for (label L=1; L < hiddenLayers.size(); ++L)
    {
        nn->push_back(
            torch::nn::Linear(hiddenLayers[L-1], hiddenLayers[L])
        );
        nn->push_back(torch::nn::GELU()); 
    }
    // - Output is 1D: value of the learned scalar field. 
    // FIXME: generalize here for vector / scalar input. TM.
    nn->push_back(
        torch::nn::Linear(hiddenLayers[hiddenLayers.size() - 1], 1)
    );
    
    // Initialize training data 
    
    // - Cell labels
    std::vector<int> cellLabels;
    std::vector<int> trainingLabels;
    std::vector<int> validationLabels;
    
    // - Training data
    torch::Tensor trainingPoints; 
    torch::Tensor trainingValues; 

    // - Validation data
    torch::Tensor validationPoints; 
    torch::Tensor validationValues; 
    
    // - Set data sizes: training a cell-centered field. 
    cellLabels.resize(mesh.nCells());
    std::iota(cellLabels.begin(), cellLabels.end(), 0);

    // Split training and validation data
    // - Use 90% from all cells for training
    std::sample
    (
        cellLabels.begin(), 
        cellLabels.end(), 
        std::back_inserter(trainingLabels),
        int(std::round(0.9 * cellLabels.size())), 
        std::mt19937(std::random_device{}())
    );
    
    // - Use 10% from all cells for validation 
    std::sample
    (
        cellLabels.begin(), 
        cellLabels.end(), 
        std::back_inserter(validationLabels),
        int(std::round(0.1 * cellLabels.size())), 
        std::mt19937(std::random_device{}())
    );
    

    // - Initialize training data 
    trainingPoints = torch::zeros({static_cast<long>(trainingLabels.size()),3});
    trainingValues = torch::zeros({static_cast<long>(trainingLabels.size()),1});

    // - Assign training cell center and vf values to tensors.
    const auto& cellcenters = vf.mesh().C();
    for(decltype(trainingLabels.size()) i = 0; i < trainingLabels.size(); ++i)
    {
        const auto cellI = trainingLabels[i];
        AI::assign(trainingPoints[i], cellcenters[cellI]);
        trainingValues[i] = vf[cellI];
    }

    // - Initialize validation data 
    validationPoints = torch::zeros({static_cast<long>(validationLabels.size()),3});
    validationValues = torch::zeros({static_cast<long>(validationLabels.size()),1});

    // - Assign validation cell center and vf values to tensors.
    for(decltype(validationLabels.size()) i = 0; i < validationLabels.size(); ++i)
    {
        const auto cellI = validationLabels[i];
        AI::assign(validationPoints[i], cellcenters[cellI]);
        validationValues[i] = vf[cellI];
    }
    
    // Train the network
    torch::optim::RMSprop optimizer( 
        nn->parameters(), 
        optimizerStep
    );

    torch::Tensor trainingPrediction = torch::zeros_like(trainingValues);
    torch::Tensor trainingMse = torch::zeros_like(trainingValues);

    torch::Tensor validationPrediction = torch::zeros_like(validationValues);
    torch::Tensor validationMse = torch::zeros_like(validationValues);
    
    size_t epoch = 1;
    double trainingMseMean = 0.,
           trainingMseMax = 0.,
           validationMseMean = 0.,
           validationMseMax = 0.;

    // - Approximate DELTA_X on unstructured meshes
    const auto& deltaCoeffs = mesh.deltaCoeffs().internalField();
    double delta_x = Foam::pow(
        Foam::min(deltaCoeffs).value(),-1
    );


    // - Open the data file for writing
    auto file_name = getAvailableFileName("pinnFoam");   
    std::ofstream dataFile (file_name);
    dataFile << "HIDDEN_LAYERS,OPTIMIZER_STEP,EPSILON,MAX_ITERATIONS,"
        << "DELTA_X,EPOCH,"
        << "TRAINING_MSE_MEAN,TRAINING_MSE_MAX,"
        << "VALIDATION_MSE_MEAN,VALIDATION_MSE_MAX\n";
     
    for (; epoch <= maxIterations; ++epoch) 
    {
        // Training
        optimizer.zero_grad();

        trainingPrediction = nn->forward(trainingPoints);
        trainingMse = mse_loss(trainingPrediction, trainingValues);

        trainingMse.backward(); 
        optimizer.step();

        trainingMseMean = trainingMse.mean().item<double>();
        trainingMseMax = trainingMse.max().item<double>();
        
        // Validation
        validationPrediction = nn->forward(validationPoints);
        validationMse = mse_loss(validationPrediction, validationValues);
        validationMseMean = validationMse.mean().item<double>(); 
        validationMseMax = validationMse.max().item<double>(); 

        std::cout << "Epoch = " << epoch << "\n"
            << "Training MSE loss mean = " << trainingMseMean << "\n"
            << "Training MSE loss max = " << trainingMseMax << "\n"
            << "sqrt(Training MSE loss mean) = " << std::sqrt(trainingMseMean) << "\n"
            << "Validation MSE loss mean = " << validationMseMean << "\n"
            << "Validation MSE loss max = " << validationMseMax << "\n"
            << "sqrt(Valiation MSE loss mean) = " << std::sqrt(validationMseMean) << "\n";  
        
        // Write the hiddenLayers_ network structure as a string-formatted python list.
        dataFile << "\"";
        for(decltype(hiddenLayers.size()) i = 0; i < hiddenLayers.size() - 1; ++i)
            dataFile << hiddenLayers[i] << ",";
        dataFile  << hiddenLayers[hiddenLayers.size() - 1] 
            << "\"" << ",";
        // Write the rest of the data. 
        dataFile << optimizerStep << "," << epsilon << "," << maxIterations << "," 
            << delta_x << "," << epoch << "," 
            << trainingMseMean << "," << trainingMseMax << "," 
            << validationMseMean << "," << validationMseMax << "\n"; 
        
        
        if (trainingMseMean < epsilon)
        {
            break;
        }
    }
    
    if (epoch >= maxIterations) 
        std::cout << "Not converged, error = " 
            << trainingMseMean << std::endl;
    
    Info<< nl;
    runTime.printExecutionTime(Info);
    
    // Network evaluation. 
    const auto& cell_centers = mesh.C(); 
    torch::Tensor input_point = torch::zeros(3);
    forAll(vf_nn, cellI)
    {
        AI::assign(input_point, cell_centers[cellI]);
        vf_nn[cellI] = nn->forward(input_point).item<double>();
    }
    vf_nn.correctBoundaryConditions();
    
    // Error calculation and output.
    error_c = Foam::mag(vf - vf_nn);
    scalar vfMaxMag = Foam::mag(Foam::max(vf)).value();
    scalar error_c_l_inf = Foam::max(error_c).value();  
    scalar error_c_l_inf_rel = error_c_l_inf / vfMaxMag;
    scalar error_c_mean_rel = Foam::average(error_c).value() / vfMaxMag;

    Info << "max(|field - field_nn|) = " << error_c_l_inf << endl; 
    Info << "max(|field - field_nn|)/|max(field)| = " 
        << error_c_l_inf_rel << endl; 
    Info << "mean(|field - field_nn|)/|max(field)| = " 
        <<  error_c_mean_rel << endl; 
    
    // Write fields
    error_c.write();
    vf_nn.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
