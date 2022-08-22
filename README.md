**DEEP VIRTUAL PION PAIR PRODUCTION**

 This is a phase-space model of $ep \rightarrow e^{\prime} p^{\prime} \pi_{1} \pi_{2}$.
 
 Here we have 8 independent kinematic variables in the final state of  $ep \rightarrow e^{\prime} p^{\prime} \pi_{1} \pi_{2}$. 
 
 Those are $Q^{2}$, $x_{B}$, $\phi_{e}$, $M_{1,2}^{2}$, $t$, $\phi_{1,2}^{*}$, $\cos\theta_{\sigma_{rest}}$ , $\phi_{\sigma_{rest}}$.
 This event generator has proceeded as follows.(with fixed incident beam energy $k=10.6 GeV$
 

   
   - Generate $x_{B}=[x_{min},x_{max}]$. For $k=10.6 GeV$, an appropriate range is
    $x_{min}=0.2$,  $x_{max}=0.8$
    
   - Generate $Q^2 \in [Q^2_{min},Q^2_{max}]$. Use $Q_{min}^{2}=1GeV^2$. Energy-momentum conservations requires (Laboratory rest frame)
   
     $k^{\prime} = \frac{k}{1+ \frac{k}{Mx_B}\left(1-\cos\theta_e\right)}$
    
     $Q^2 = \frac{2k^2(1-\cos\theta_e)}{1 + \frac{k}{Mx_B}\left(1-\cos\theta_e\right)}$
   
     $Q^2_\text{max} = 4k^2\left/\left[1 + \frac{2k}{Mx_B}\right] \right.  = \frac{4k^2 Mx_B }{M x_B + 2k}$
    
     Note: $Q^2_{max}$ is a function of $x_{B}$. Checked that $k'$ is physical: reject events with $k'<0$ or $k'>k$.           Calculated the invariant $W^{2}=M^{2}+Q^{2}(\frac{1}{x_{B}}-1)$: reject events with $W^{2}-M^{2}< 4m\pi^{2}$. Use the   charged pion mass $m_{\pi}=0.139 GeV$
 
   - Generate $\phi_{e} \in [0,2\pi]$. 
     
   - Generate the $\pi\pi$ invariant mass squared
    
      $M_{1,2}^2 \in [4m_{\pi}^{2}, (\sqrt[]{W^{2}}-M)^{2}]$
    
   - Generate  $t \in [t_{max},t_{min}]$.
 
   - Generate $\phi_{1,2}^\ast \in[0,2\pi]$.

   - Generate $\cos\theta_{\sigma-\text{Rest}} \in [-1,1]$.

   - Generate $\phi_{\sigma-\text{Rest}} \in [0,2\pi]$.
Compute the $\sigma\text{-Rest}$ frame four-vectors $p_1$ and $p_2$ of the two pions.

    
Moreover, Raditive Correction has added to this generator.

**RUN THE GENEARATOR**
   *Local Machine
 Pre-requisites -ROOT Cern
 
 1 Install ROOT CERN
 
 2 gitclone of the event generator
  https://github.com/JeffersonLab/deep-pipi-gen.git
 
 3 Run "make". 
   this creates the "deep-pipi-gen" file(executable file)
 
 4 Then Run as follows
    ./deep-pipi-gen --trig 10000 
         Seeds are optional
 
 
 

