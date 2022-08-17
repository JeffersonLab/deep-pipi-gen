//
//  Monte_RC.cpp
//
//
//  Created by Dilini Bulumulla on 8/23/19.
//  References:
//  exclurad
//  A. Afanasev, I. Akushevich, V. Burkert, K. Joo
//     PHYSICAL REVIEW D 66, 074004 (2002)
//  diffrad
//  I. Akushevich, Eur. Phys. J. C 8, 457–463 (1999)

#include "Deep_pipi.hpp"

int main(int argc, char *argv[]) {
  
    // choose a random number seed from system clock:
    //auto time = std::chrono::system_clock::now();
    //std::chrono::microseconds ms =
    //    std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()); 
    //long long unsigned seed_value = ms.count(); 

    char* short_options = (char*)"a:b:c:";
    const struct option long_options[] = {
                {"trig",required_argument,NULL,'a'},
                {"docker",optional_argument,NULL,'b'},
                {"seed",optional_argument,NULL,'c'}
    };

    int nevents,seed;
    for (int i=1; i<argc; i+=2) {
        TString key = argv[i];
        TString val = argv[i+1];
        if (key.EqualTo("--trig")) {
            nevents = val.Atoi();
        }
        else if (key.EqualTo("--seed")) {
            seed = val.Atoi();
            gRandom->SetSeed(seed);
        }
    }

    FILE *outfile;
    outfile = fopen("deep-pipi-gen.dat", "w");  // using relative path name of file
    if (outfile == NULL) {
        printf("Unable to open file.");
    }
    Double_t Q2,xBj,nu,kprime,theta,csthe,u,x,y,W2,P12CM,EpprimeCM,csthe12, snthe12, M122,E12CM,EpCM,q0CM,phie,qCM,PCM,phi12,thetP12CM,t,theg,phig,cstheg;
    Double_t csth1Rest, th1Rest, phi1Rest, ppiRest,vmaxD,vmaxH,vmaxE;
    const Double_t Mtarget=0.93827;
    const Double_t M2=Mtarget*Mtarget;
    const Double_t kBeam=10.6;
    const Double_t Mpi=0.13957;
    const Double_t Mpi2=Mpi*Mpi;
    const Double_t mElectron=0.00050;
    const Double_t ZTarget=1.0;
    const Double_t ATarget=1.0;
    const Double_t tFrac = 1.0 ; // Fraction of "t" phase space
    const Double_t tMaxGen0 = -3.0; // alternate maximum negative value of t
    const Double_t M122Limit = 1.2; // GeV2, Maximal sigma mass
    const Double_t alphaQED = 1.0/137.03;
    const Double_t mElSq    = mElectron*mElectron;
    const Double_t pi       = acos(-1.0);
    const Double_t tmom = 0.0;
    const Double_t slope=5;
    Double_t mLepton = mElectron;
    

    Double_t Q2min=1.0, Q2max=10, xmin=0.1, xmax=0.8, M122min=4.*Mpi2, M122max,tmin=0,tmax=1;
    const Double_t rad2deg=180./acos(-1.0);
    double xVert=0.0, yVert=0.0, zVert=-1.94;
    double delta_inf,xBj_bar,SS_bar;
    double vcut,lm,Sprime,Xprime,vmax,tt1,tt2,tq,SS,XX,VV1,aa1,aa2,VV2,ys,xs,Sp,Sx,St;
    double lambdaq,bb1,bb2,bb;
    double delvr,delvacl, delvach,sigmaobs,sigmaf,delta_s,RF,v1,v2,Q2_bar,W2_bar, post_rad,pre_rad,vmiss;
    double u1,u2,lambda1,lambda2,lamdaW,lambda,lambdas,E1,E2,Eh,Ep,Eq,p1,p2,pp, csthe1,csthe2,sinthe1,sinthe2,ph,kPrMag;
    double tmaxGen;
    double psf;
    TH1D *hQ2 = new TH1D("hQ2","Q^{2} Distribution  ;  Q^{2} (GeV^{2})  ; Events  ",
                         100, 0.0, Q2max);
    hQ2->Sumw2();
    TH2D *hxQ2 = new TH2D("hxQ2", "Q^{2} vs. x_{Bj}  ;   x_{Bj}   ; Q^{2}   ", 40, 0.,1., 40,0.,Q2max);
    TH1D *hDPx = new TH1D("hDPx", "Final State P_{x} - Initial State P_{x}; #Delta P_{x}   ;  Events  ",
                          100, -0.25,0.25);
    // TH1D *hDPy = new TH1D("hDPy", "Final State P_{y} - Initial State P_{y}; #Delta P_{y}   ;  Events  ",
    // 95, -9.5,9.5);
    TH1D *h_RF = new TH1D("h_RF", "RF  ;  Events  ",95, 0.0,4.0);
    TH1D *h_sigmaobs = new TH1D("h_sigmaobs", "RF  ;  Events  ",95, 0.0,4.0);
    const int n_rad_bin=200;
    TH1D *h_rad = new TH1D("h_rad",
                           "(pre_rad+post_rad)/v_{max}; (GeV)^{2}  ;  Events  ",
                           n_rad_bin, -0.10,0.90);
    TH1D *hDP0 = new TH1D("hDP0", "Final State E - Initial State E; #Delta E  ;  Events  ",
                          95, -9.5,9.5);
    TH1D *hvMax = new TH1D("hvMax"," v_{Max} ; #Lambda^{2}-M^{2} (GeV^{2})  ;Events   ",
                           100, 0.0,2.50);
    TH1D *ht = new TH1D("ht"," t = #Delta^{2} (GeV^{2})  ;Events     ",
                        100, -2.0, 0.0);
    TH1D *h_Lambda = new TH1D("h_Lambda"," LambdaSq;LambdaSq  ;Events     ",
                        100, -1.0, 3.0);
    
    int npartf = 4+2;   // e' + p' + piplus+ piminus + k_pre + p_post
    int nTrack;
    int active = 1;
    int inactive = 0;
    double kPolar=1;
    double TPolar=1;
    int pidElectron=11;
    int pidProton  =2212;
    int pidPiPlus  =211;
    int pidPiminus =-211;
    int pidPhoton =22;
    int parent=0, daughter=0;
    kprime=y;
    theta=x;
    TRandom3 ran3;
    TVector3 unitzq, unitxq, unityq, unitz12, unitx12, unity12,unitx,unity,unitz,unitpi0x,unitpi0y,unitpi0z;
    TVector3 kBeam3, kScat3, qVector3, BoostLab, BoostRest, pi0Boost;
    TVector3 p3Pion1,PVec12CM,p3pi0,p3gammaN;
    TLorentzVector k4BeamLab, k4ScatLab, P4Total, P4Vec12, P4PprimeCM, P4pion1Rest, P4pion2Rest;
    TLorentzVector k4BeamCM, k4ScatCM;
    TLorentzVector P4gamma1Lab,P4gamma2Lab, P4gamma3Lab,P4gamma4Lab;
    TLorentzVector P4PprimeLab, P4pion1Lab, P4pion2Lab, P4TotFinal;
    TLorentzVector P4TargetLab,k4PreRad,k4PostRad;
    P4TargetLab.SetPxPyPzE(0.0, 0.0, 0.0, Mtarget);
    
    kBeam3.SetXYZ(0.,0.,kBeam);
    unitz.SetXYZ(0.,0.,1.);
    unitx.SetXYZ(1.,0.,0.);
    unity.SetXYZ(0.,1.,0.);
    k4BeamLab.SetVectM(kBeam3,mLepton);
    
    Int_t nEvt=0;
    while (nEvt<nevents) {
        Q2=(Q2max-Q2min)*(ran3.Uniform())+Q2min;
        xBj=(xmax-xmin)*(ran3.Uniform())+xmin;
        nu=Q2/(2*xBj*Mtarget);
        if (nu > (kBeam-mLepton)) continue;
        kprime=kBeam-nu;
        kPrMag= sqrt(kprime-mLepton)*sqrt(kprime+mLepton);
        // -Q2 = 2m_e^2 - 2k\cdot k'
        // -Q2 = 2m_e^2 - 2E_1 E_2 + 2*kvec1*kvec2*csthe
        csthe = ( 2.*kprime*k4BeamLab.E()-(Q2+2.*mLepton*mLepton) ) /(2.*kPrMag*k4BeamLab.P());
        if (csthe*csthe>1.0) continue;
        theta=acos(csthe);
        phie = 2.*acos(-1.0)*ran3.Uniform();
        kScat3.SetXYZ(kPrMag*sin(theta)*cos(phie), kPrMag*sin(theta)*sin(phie),kPrMag*csthe);
        k4ScatLab.SetVectM(kScat3,mLepton);
        theta =rad2deg;
        qVector3 = kBeam3-kScat3;
        unitzq = qVector3.Unit();
        unityq = kBeam3.Cross(kScat3);
        unityq = unityq.Unit();
        unitxq = unityq.Cross(unitzq);
        W2=M2+Q2*(1./(xBj)-1);
        if (W2 < W2Min) continue;
        if (W2-M2-4.*Mpi2<2.*Mtarget*Mpi) continue;
        hQ2->Fill(Q2);
        hxQ2->Fill(xBj,Q2);
        //  Corrected CEH 7 Sept 2019
        //M122max=TMath::Min(pow(sqrt(W2)-Mtarget,2),sqrt(M122Limit) );
        M122max=TMath::Min(pow(sqrt(W2)-Mtarget,2),M122Limit );
        M122=(M122max-M122min)*(ran3.Uniform())+M122min;
        //printf("values of W2= %10.3f \n",W2);
        q0CM = (W2-M2-Q2)/(2.*sqrt(W2));
        qCM  = sqrt(q0CM*q0CM +Q2);
        PCM = qCM;
        E12CM = (W2-M2+M122)/(2.*sqrt(W2));
        EpCM  = (W2+M2+Q2)/(2.*sqrt(W2));
        EpprimeCM=(W2+M2-M122)/(2.*sqrt(W2));
        if (E12CM*E12CM < M122) {
            printf("Event %d, E12CM, M122 = %7.3f, %7.3f \n",
                   nEvt, E12CM, M122);
            continue;
        }
        P12CM=sqrt(E12CM*E12CM-M122);
        
        tmax = 2.*M2 - 2.*EpCM*EpprimeCM - 2.*PCM*P12CM; // Max NEGATIVE
        tmin = 2.*M2 - 2.*EpCM*EpprimeCM + 2.*PCM*P12CM; // Min NEGATIVE
        tmaxGen = std::max(tmax,tMaxGen0);
        t    = (tmaxGen-tmin)*ran3.Uniform()*tFrac + tmin;
        ht->Fill(t);
        //  For any t, there is a range of radiation values v_cut allowed between 0 and v_{Max}
        //  calculate mesonic final state kinematics after radiation generation.
        //
        //  These variables must be recalculated after radiation generation
        csthe12=(t-2.*M2+2.*EpCM*EpprimeCM)/(2.*PCM*P12CM); //cos(theta_h) in Diffrad
        if (csthe12*csthe12> 1.0){
          //  printf("Event %d, Invalid On-shell kinematics, csthe12 = %7.4f \n",i, csthe12);
            continue;
        }
        snthe12 = sqrt(1.-csthe12*csthe12);
        phi12 = 2.*acos(-1.0)*ran3.Uniform(); //phi12
        //  fill P4PprimeLab??
        //calculate psf-phase space factor
        psf = (Q2max-Q2min)*(xmax-xmin)*2.*pi * (M122max-M122min)*(tmaxGen-tmin)*4.*pi;
        
        
  //Diffrad
        //  On-shell (no radiation) values
        tt1=W2-Q2-M2;
        tt2=W2-M2+M122;
        tq =Q2+t-M122;
        
        lm=TMath::Log(Q2/mElSq);
        vmaxD=tt2+0.5/Q2*(-tt1*tq-sqrt(tt1*tt1+4.0*Q2*W2)*sqrt(tq*tq+4.0*(M122)*Q2)); // Diffrad definition of vmax
        vmaxH=((Q2-t+M122)/(2*xBj)+t) * (1-sqrt(1-W2*(tmax-t)*(t-tmin)/Q2/pow((Q2-t+M122)/(2*xBj)+t,2))); //Dr.Hyde's definition of vmax
        vmaxE= pow(sqrt(W2)-sqrt(M122),2)-Mtarget*Mtarget; //exclurad definition
        if((vmaxD-vmaxH)>1.e-6||(vmaxD-vmaxH)<-1.e-6) {
            printf("Event %d, vmax D,H = %9.6f, %9.6f  GeV2 \n ", nEvt, vmaxD, vmaxH);
            continue;
        }
            vmax= vmaxD;
        /*
        if((vmaxE-vmaxH)>1.e-6||(vmaxE-vmaxH)<-1.e-6){
            printf("Event %d, vmax E,H = %9.6f, %9.6f  GeV2 \n ", i, vmaxE, vmaxH);
            vmax = vmaxE;
        }
         */
            hvMax->Fill(vmax);
        
       
        SS=2*kBeam*Mtarget;
        ys=(W2+Q2-M2)/SS;    //Diffrad
        xs=Q2/(ys*SS);       //Diffrad eq.10
        XX=(1-ys)*SS;        //Diffrad eq.10
        
        Sp=SS+XX;            //Diffrad eq.10
        Sx=SS-XX;            //Diffrad eq.10
        St=Sx+t;             //Diffrad eq.10
       
       
       // Diffrad definition of vv1 and vv2
/*        aa1=(Q2*Sp*St-(SS*Sx+2*M2*Q2)*tq)/2.0/lambdaq;
        aa2=(Q2*Sp*St-(XX*Sx-2*M2*Q2)*tq)/2.0/lambdaq;
         printf("vmax=%10.6f \n",aa2);
        bb1=sqrt((Q2*St*St)-(St*Sx*tq)-(M2*tq*tq)-(M122*lambdaq));
          printf("vmax=%10.6f \n",bb1);
        bb2=sqrt((SS*XX*Q2)-(M2*Q2*Q2)-(mElSq*lambdaq));
        bb=bb1*bb2/lambdaq;
        
        VV1=(aa1+bb*cos(phi12))*2.0;
        VV2=(aa2+bb*cos(phi12))*2.0;
        printf("ssh=%10.6f \n",VV1); */
        
        //  Electron scattering variables, independent of final state radiation
        // in gamma*+P CM frae
        u1=SS-Q2;                   //exclurad eq. 2
        u2=XX+Q2;                   //exclurad eq. 2
        lambda1=u1*u1 - 4.*mElSq*W2;  //exclurad eq. 2
        lambda2=u2*u2 - 4.*mElSq*W2;  //exclurad eq. 2
        lambdas=(SS*SS)-4.*mElSq*M2;  //exclurad eq. 2
        lambdaq =Sx*Sx+4.*M2*Q2;      //exclurad eq. 2
        lambda=Q2*u1*u2-(Q2*Q2*W2)-mElSq*lambdaq;  //exclurad eq. 2
        E1=u1/(2.*sqrt(W2));       //exclurad eq. 3
        E2=u2/(2.*sqrt(W2));       //exclurad eq. 3
        Eq=(Sx-(2.*Q2))/(2.*sqrt(W2));  //exclurad eq. 3
        Ep=(Sx+2.*M2)/(2.*sqrt(W2));    //exclurad eq. 3
        p1=sqrt(lambda1)/(2.*sqrt(W2));  //exclurad eq. 3
        p2=sqrt(lambda2)/(2.*sqrt(W2));  //exclurad eq. 3
        pp=sqrt(lambdaq)/(2.*sqrt(W2));  //exclurad eq. 3
        csthe1=((u1*(Sx-2.*Q2))+2.*Q2*W2)/sqrt(lambda1)/sqrt(lambdaq);  //exclurad eq. 3
        csthe2=((u2*(Sx-2.*Q2))-2.*Q2*W2)/sqrt(lambda2)/sqrt(lambdaq);  //exclurad eq. 3
        sinthe1=2.*sqrt(W2)*sqrt(lambda)/sqrt(lambda1)/sqrt(lambdaq);  //exclurad eq. 3
        sinthe2=2.*sqrt(W2)*sqrt(lambda)/sqrt(lambda2)/sqrt(lambdaq);  //exclurad eq. 3
        Double_t mh2=M122;
        Double_t mu=Mtarget;
        if(sqrt(W2) < sqrt(mh2)+mu){
            printf("Didn't we already test W2 = %7.4f \n", W2);
            continue;
        }
        // Corrected parenthesis error CEH 7 Sept 2019
        //Eh=W2+mh2-(mu*mu)/(2.*sqrt(W2));   //H(e,e' pi+ pi-) X channel for now:  m_h = m_\pi\pi m_u = mProton  ////exclurad eq. 4
        //  Hadronic final state-dependent (q+P CM frame)
        Eh=(W2+mh2-mu*mu)/(2.*sqrt(W2));
        //       vmiss=W2+mh2-(mu*mu)-2*sqrt(W2)*Eh; //exclurad eq. 24
        lamdaW=pow((W2+mh2-mu*mu),2)-4.*mh2*W2;

        ph=sqrt(lamdaW)/(2.*sqrt(W2)); //exclurad eq. 26
       
        VV1=2.*(E1*Eh-p1*ph*(csthe12*csthe1+snthe12*sinthe1*cos(phi12))); //exclurad eq. 26
        VV2=2.*(E2*Eh-p2*ph*(csthe12*csthe2+snthe12*sinthe2*cos(phi12))); //exclurad eq. 26
        
        
        Sprime= SS-Q2-VV1;                                              //Diffrad 9
        Xprime= XX+Q2-VV2;                                              //Diffrad 9
        if (Sprime<= 0.0 || Xprime<= 0.0){
            printf("Event %d, Sprime, Xprime %7.4f, %7.4f Negative invalid \n",
                   nEvt, Sprime,Xprime);
            printf("     SS, XX, VV1, VV2 = %7.3f, %7.3f, %7.3f, %7.3f \n",
                   SS, XX, VV1, VV2);
            printf("     CM E1, p1, E2, p2, Eh, ph = %7.3f, %7.3f, %7.3f, %7.3f %7.3f %7.3f \n",
                   E1, p1, E2, p2, Eh, ph);
            printf("     CM csth1, sinthe1 = %7.3f, %7.3f, %7.3f, %7.3f \n",
                   csthe1, sinthe1, csthe2, sinthe2);
            continue;
        }
        delta_inf= alphaQED/pi* (lm-1.0)*TMath::Log(vmax*vmax/(Sprime*Xprime)); // Diffrad 13
        delvr = alphaQED/pi * (1.5*lm-2.0-0.5*log(Xprime/Sprime)*log(Xprime/Sprime)+TMath::DiLog(1.-(Q2*M2/(Sprime*Xprime)))-(pi*pi/6.0)); //Diffrad 13
        delvacl=2.*alphaQED*(-5./9.+lm/3.)/pi;
        delta_s=alphaQED*(log(Q2/mElSq) - 1.0)/pi;
        RF= pow((vmax*vmax/(Sprime*Xprime)),delta_s)*(1+delvr+delvacl);
        h_RF->Fill(RF);
        v1= 0.0; //vmax*(ran3.Uniform());
        v2 =0.0; // (vmax-v1)*(ran3.Uniform());
        pre_rad=vmax* pow(ran3.Uniform(),1./delta_s);
        post_rad=(vmax-pre_rad)*pow(ran3.Uniform(),1./delta_s);
        // Recalculate Sprime, Xprime
        //  Recalculate V1,2
        //  Post radiation \Lambda^2 = mu^2 + pre_rad + post_rad
        // The "bar" values
        Eh = (W2-mu*mu-pre_rad-post_rad+M122)/(2.*sqrt(W2));
        if (Eh<sqrt(M122)){
            printf("Event %d, Invalid radiative kinematics E12CM, M_pi,pi = %7.4f, %7.4f \n",
                   nEvt, Eh,sqrt(M122));
            continue;
        }
        ph = sqrt(Eh*Eh-M122);
        // q_vector^{CM} = sqrt(Eq*Eq+Q2)
        csthe12 = (Q2+t + 2.*Eq*Eh - M122)/(2.*sqrt(Eq*Eq+Q2)*ph);
        if (csthe12*csthe12>1.0){
            printf("Event %d, csthe12=%10.3f gamma* P CM frame\n",
                   nEvt,csthe12);
            printf("     tMin, t = %7.4f, %7.4f \n", tmin, t);
            continue;}
        thetP12CM = acos(csthe12);
        snthe12   = sin(thetP12CM);
        
        // exclurad eq. 26:   Radiative kinematics
        VV1=2.*(E1*Eh-p1*ph*(csthe12*csthe1+snthe12*sinthe1*cos(phi12)));
        VV2=2.*(E2*Eh-p2*ph*(csthe12*csthe2+snthe12*sinthe2*cos(phi12)));
        
        Sprime= SS-Q2-VV1;                                              //Diffrad 9
        Xprime= XX+Q2-VV2;                                              //Diffrad 9

        k4PreRad=(pre_rad/Sprime)*k4BeamLab;
        k4PostRad=(post_rad/Xprime)*k4ScatLab;
        //h_rad->Fill((pre_rad/Sprime+post_rad/Xprime));
        //h_rad->Fill(k4PreRad.E()+k4PostRad.E());
        
        
        
        //sigmaf is the approximate sigma_f/sigmaborn
        sigmaf=2.*alphaQED*slope*vmax*(lm-1.)*(Q2+M122)/(Sx-Q2-M122)/pi;
  //      printf("sigmaf/sigmaborn=%10.6f \n",sigmaf);
        //sigmaobs is the sigmaobs/sigmaborn
        sigmaobs=TMath::Exp(delta_inf)*(1+delvr+delvacl)+sigmaf;
  //      printf("sigmaobs/sigmaborn=%10.6f \n",sigmaobs);
        h_sigmaobs->Fill(sigmaobs);
        
        E12CM = Eh;
        P12CM = ph;
   /*     //Calculation of born cross section at vertex kinematics
        TLorentzVector q_bar = k4BeamLab - k4PreRad - k4PostRad - k4ScatLab;
        Q2_bar= - q_bar.M2();
        W2_bar = Mtarget*Mtarget + 2.0*(q_bar.Dot(p4TargetLab)) - Q2_bar;
        xBj_bar= Q2_bar/(2.0*(q_bar.Dot(p4TargetLab)));
        SS_bar = SS - 2.0*(k4PreRad.Dot(p4TargetLab));
        //p4primeLab += (-1.0)*(k4PreRad+k4PostRad);
        //Redefine energy and 3 momentum of  "hadron = pi pi" due to "bar" kinematics
        //  New p4pipi defined by q_bar, P4Target, t, and phi_h ?? */
        //  Inconsistent, Don't use bar variables to define CM or final state.
        //  bar variables are only for vertex
        TLorentzVector q4_bar = k4BeamLab - k4PreRad - k4PostRad - k4ScatLab;
        TLorentzVector k4Beam_bar = k4BeamLab - k4PreRad;
        TLorentzVector k4Scat_bar = k4ScatLab + k4PostRad;
        TVector3 kBeam3_bar = k4Beam_bar.Vect();
        TVector3 kScat3_bar = k4Scat_bar.Vect();
        TVector3 qBar3 = q4_bar.Vect();
        /*
        unitzq = qBar3.Unit();
        unityq = kBeam3_bar.Cross(kScat3_bar);
        unityq = unityq.Unit();
        unitxq = unityq.Cross(unitzq);
         
        Q2_bar= - q4_bar.M2();
        W2_bar = Mtarget*Mtarget + 2.0*(q4_bar.Dot(P4TargetLab)) - Q2_bar;
        xBj_bar= Q2_bar/(2.0*(q4_bar.Dot(P4TargetLab)));
        SS_bar = SS - 2.0*(k4PreRad.Dot(P4TargetLab));
*/
        // Redefine CM values for e p --> e \Lambda \pi\pi
        // v1, v2 were incorrectly defined:  CEH 9/10/19
        //double LambdaSq = (mu*mu) + v1 + v2; // replace M2 with M_u^2?
        double LambdaSq = (mu*mu) + pre_rad + post_rad;
        h_Lambda->Fill(LambdaSq);
        /*
        q0CM = (W2_bar-M2-Q2_bar)/(2.*sqrt(W2_bar));
        qCM  = sqrt(q0CM*q0CM +Q2_bar);
        PCM = qCM;
        E12CM = Eh;
        // Correction CEH 7 Sept 2019
        if (E12CM<sqrt(M122)){
            printf("Event %d, Invalid radiative kinematics E12CM, M_pi,pi = %7.4f, %7.4f \n",
                   i, E12CM,sqrt(M122));
            printf("Revised E12CM = (W2-M2-pre_rad-post_rad+M122)/(2W) = %7.4f \n",
                   (W2-M2-pre_rad-post_rad+M122)/(2.*sqrt(W2)));
            continue;
        }
        EpCM=(W2_bar+M2+Q2_bar)/(2.*sqrt(W2_bar));
        EpprimeCM=(W2_bar+M2-M122)/(2.*sqrt(W2_bar));
        if (EpprimeCM < Mtarget) {
            printf("Event %d, Invalid Radiative kinematics EpprimeCM = %7.3f \n",
                   i, EpprimeCM);
            continue;
        }
         */
    
// Define final state vectors
        // unitz12.SetXYZ(sin(thetP12)*cos(phi12),sin(thetP12)*sin(phi12),csthe12);
        //  Define unitz12 = direction of pi pi pair in CM (opposite recoil proton in CM)
        unitz12 = csthe12*unitzq + snthe12*(cos(phi12)*unitxq + sin(phi12)*unityq);
        unitx12 = -unitzq + unitz12*(unitzq.Dot(unitz12));
        unitx12 = unitx12.Unit();
        unity12 = unitz12.Cross(unitx12);
        // P4Vec12CM sigma meson 4-vector in gamma-P CM frame
        //  Redefine after radiation calculation
        PVec12CM = P12CM*unitz12;
        P4Vec12.SetVectM(PVec12CM,sqrt(M122));
        //P4PprimeCM.SetVectM(-PVec12CM,Mtarget); // not valid with radiation
        BoostRest = P4Vec12.BoostVector();  //  Define boost 3-vector from pi pi rest frame back to gamma-P CM frame
        
        csth1Rest = 2.*ran3.Uniform()-1.0;
        th1Rest   = acos(csth1Rest);
        phi1Rest  = 2.*acos(-1.)*ran3.Uniform();
        ppiRest   = sqrt(M122/4.0-Mpi2);
        p3Pion1=ppiRest*(csth1Rest*unitz12
                         +sin(th1Rest)*(cos(phi1Rest)*unitx12
                                        +sin(phi1Rest)*unity12));
        P4pion1Rest.SetVectM(p3Pion1,Mpi);
        p3Pion1  *= -1.0;
        P4pion2Rest.SetVectM(p3Pion1,Mpi);
        //  Boost from sigma-meason Rest frame to CM frame
        P4pion1Rest.Boost(BoostRest);
        P4pion2Rest.Boost(BoostRest);
        P4Total.SetVectM(qVector3, sqrt(W2));
        //printf(" not radiative kinematics P4Total = %7.3f \n", P4Total.Px());
        //PTotal += P4TargetLab;
        BoostLab = P4Total.BoostVector();
        // P4Vec12.Boost(-BoostLab);    // sigma meson 4-vector in lab frame
        P4pion1Lab = P4pion1Rest;
        P4pion2Lab = P4pion2Rest;
        P4pion1Lab.Boost(BoostLab);
        P4pion2Lab.Boost(BoostLab);
        //P4PprimeLab = P4PprimeCM;
        //P4PprimeLab.Boost(BoostLab);
        
        //P4TotFinal   = P4PprimeLab + P4pion1Lab + P4pion2Lab +k4PostRad+k4PreRad;
        P4PprimeLab = P4Total- P4pion1Lab - P4pion2Lab - k4PostRad-k4PreRad;
        //hDPx->Fill(P4TotFinal.Px()-P4Total.Px());
        hDPx->Fill(P4PprimeLab.M2()-Mtarget*Mtarget);
        h_rad->Fill(2.*P4PprimeLab.Dot(k4PreRad) +2.*P4PprimeLab.Dot(k4PostRad) +2.*k4PostRad.Dot(k4PreRad));
        P4PprimeLab.SetE(sqrt(P4PprimeLab.P()*P4PprimeLab.P()+Mtarget*Mtarget));
        
        if(k4PreRad.E()<0.0 || k4PostRad.E()<0.0){
            printf("Event %d, negative energy photon PreE, PostE = %7.4f, %7.4f \n",
                   nEvt, k4PreRad.E(),k4PostRad.E());
            continue;
        }
        
        nTrack = 0;
        fprintf(outfile," %5d %5.1f %5.1f %5.1f %5.1f  %10.4f %10.4f %10.4f %10.4f %10.4f %10.f\n", npartf,
                ATarget, ZTarget, kPolar, TPolar, xBj, nu/kBeam, sqrt(W2), Q2, RF,psf);
        nTrack++;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f %10.4f \n",
                nTrack, -1.0, active, pidElectron, 0, 0,
                kScat3.Px(), kScat3.Py(), kScat3.Pz(), sqrt(kprime*kprime+mElectron*mElectron), mElectron,
                xVert, yVert, zVert); // scattered electron
        nTrack++;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f %10.4f \n",
                nTrack,  1.0, active, pidProton, 0, 0,
                P4PprimeLab.Px(), P4PprimeLab.Py(), P4PprimeLab.Pz(), P4PprimeLab.E(), P4PprimeLab.M(),
                xVert, yVert, zVert); // Recoil proton
        nTrack++;
        // parent = 0;
        // daughter = nTrack+1;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f \n",
                nTrack,  1.0, active, pidPiPlus, parent, daughter,
                P4pion1Lab.Px(), P4pion1Lab.Py(), P4pion1Lab.Pz(), P4pion1Lab.E(), P4pion1Lab.M(),
                xVert, yVert, zVert); // Piplus
        
        nTrack++;
        // parent= 0;
        // daughter=nTrack+1;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f \n",
                nTrack,  -1.0, active, pidPiminus, parent, daughter,
                P4pion2Lab.Px(), P4pion2Lab.Py(), P4pion2Lab.Pz(), P4pion2Lab.E(), P4pion2Lab.M(),
                xVert, yVert, zVert); // Piminus
        
        nTrack++;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f \n",
                nTrack,  0.0, active, pidPhoton, parent, daughter,
                k4PreRad.Px(), k4PreRad.Py(), k4PreRad.Pz(), k4PreRad.E(), k4PreRad.M(),
                xVert, yVert, zVert); // pre_rad
        
        nTrack++;
        fprintf(outfile," %5d %5.1f %5d %5d %5d %5d %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f \n",
                nTrack,  0.0, active, pidPhoton, parent, daughter,
                k4PostRad.Px(), k4PostRad.Py(), k4PostRad.Pz(), k4PostRad.E(), k4PostRad.M(),
                xVert, yVert, zVert); // post-rad
        
        
        
        
        nEvt++;
    }
    fclose(outfile);
    
    TCanvas *c1 = new TCanvas("c1","1D",50,100,600,400);
    TCanvas *c2 = new TCanvas("c2", "2D", 100,50,600,400);
    TCanvas *c3 = new TCanvas("c3","1D",75,75,600,400);
    TCanvas *c4 = new TCanvas("c4","1D",75,100,600,400);
    TCanvas *c5 = new TCanvas("c5","1D",75,125,600,400);
    TCanvas *c6 = new TCanvas("c6","1D",75,125,600,400);
    
    c1->cd(0);
    c1->SetLogy();
    //hQ2->Draw();
    //hQ2->SetLineColor(kRed);
    //ht->Draw();
    hvMax->Draw();
    
    
    c2->cd(0);
    hxQ2->Draw();
    hxQ2->SetLineColor(kRed);
    
    c3->cd(0);
    hDPx->SetTitle("Final Proton MassSq - M^{2}");
    hDPx->Draw();
    
    
    
    c4->cd(0);
    h_RF->Draw();
    h_sigmaobs->SetLineColor(kRed);
    h_sigmaobs->Draw("same");
    
    c6->cd(0);
    h_Lambda->Draw();
    
    
    
    c5->cd(0);
    c5->SetLogy();
    h_rad->Draw();
 //   gStyle->SetOptFit(111111);
    TF1 *f_rad = new TF1("f_rad","[0]*[1]*pow(x,-1.+[1])",0.0001,0.9);
    f_rad->SetNpx(2*n_rad_bin);
    f_rad->SetParameter(0,h_rad->GetSum()/n_rad_bin);
    f_rad->SetParameter(1,0.07);
    f_rad->SetParName(1,"#delta");
    //h_rad->Fit(f_rad,"I");
    //f_rad->SetLineStyle(2);
    //f_rad->DrawCopy("same");
    f_rad->SetLineStyle(1);
    h_rad->Fit(f_rad,"IR");
    TLatex tl;
    tl.SetTextAlign(22);
    tl.DrawLatex(0.5,h_rad->GetSum()/10.,"#frac{Sum}{Bins}#delta x^{-1+#delta}");
    return nEvt;
    
    
}




