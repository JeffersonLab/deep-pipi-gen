//
//  Monte_RC.hpp
//  
//
//  Created by Dilini Bulumulla on 8/23/19.
//

#ifndef Monte_RC_hpp
#define Monte_RC_hpp

#include <getopt.h>
#include <stdio.h>
#include <chrono>

#include <TCanvas.h>
#include <TH2D.h>
#include <TRandom3.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TF1.h>

#include <stdio.h>
const Double_t alphaQED = 1.0/137.03;
const Double_t mElectron= 0.511e-3;
const Double_t mElSq    = mElectron*mElectron;
const Double_t MProton  = 0.938;
const Double_t MPrSq    = MProton*MProton;
const Double_t MRho    = 0.768;
const Double_t MRhoSq    = MRho*MRho;
const Double_t pi       = acos(-1.0);
const Double_t rad2deg=180./acos(-1.0);

//const Double_t Q2 = 4.0;
const Double_t W2Min = 4.00;
const Double_t bmom = 10.6;
const Double_t tmom = 0.0;
const Double_t tdif=-0.5;
//const Double_t xBj   = 0.48;
const Double_t slope=5;

//  Hadronic Vacuum Polarization parameters
const Double_t aaa = -1.345e-9;
const Double_t bbb = -2.302e-3;
const Double_t ccc = 4.091;


#endif /* Monte_RC_hpp */
