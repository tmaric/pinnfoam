    volScalarField vf 
    (
        IOobject
        (
            "psi_c",
            runTime.timeName(), 
            mesh, 
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volScalarField vf_nn (vf.name() + "_nn", vf);

    volScalarField error_c 
    (
        IOobject
        (
            "error_c", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedScalar("error_c", vf.dimensions(), 0)
    );
