**DEEP VIRTUAL TWO PION PAIR PRODUCTION**

 This is a phase-space model of $ep \rightarrow e^{\prime} p^{\prime} \pi_{1} \pi_{2}$.
 
 Here we have 8 independent kinematic variables in the final state of  $ep \rightarrow e^{\prime} p^{\prime} \pi_{1} \pi_{2}$. 
 
 Those are $Q^{2}$, $x_{B}$, $\phi_{e}$, $M_{1,2}^{2}$, $t$, $\phi_{1,2}^{*}$, $\cos\theta_{\sigma_{rest}}$ , $\phi_{\sigma_{rest}}$.
 This event generator has proceeded as follows.(with fixed incident beam energy $k=10.6 GeV$
 

   
   - Generate $x_{B}=[x_{min},x_{max}]$. For $k=10.6 GeV$, an appropriate range is
    $x_{min}=0.2$,  $x_{max}=0.8$
    
   - Generate $Q^{2} \in [Q^{2}_{min},Q^{2}_{max}]$. Use $Q_{min}^{2}=1GeV^2$. Energy-momentum conservations requires (Laboratory rest frame)
   
     $k^{\prime} = \frac{k}{1+ \frac{k}{Mx_B}\left(1-\cos\theta_e\right)}$
    
     $Q^2 = \frac{2k^2(1-\cos\theta_e)}{1 + \frac{k}{Mx_B}\left(1-\cos\theta_e\right)}$
   
     $Q^2_\text{max} = 4k^2\left/\left[1 + \frac{2k}{Mx_B}\right] \right.  = \frac{4k^2 Mx_B }{M x_B + 2k}$
    
 
 Note that $Q^{2}_{max}$ is a function of $x_{B}$. Check that $k'$ is physical: reject events with $k'<0$ or $k'>k$. Calculate the invariant $W^{2}=M^{2}+Q^{2}(\frac{1}{x_{B}}-1)$: reject events with $W^{2}-M^{2}< 4m\pi^{2}$. Use the charged pion mass $m_{\pi}=0.139 GeV$ and neutral pion mass $m_{\pi}=0.135 GeV$
    
    \item Generate $\phi_{e} \in [0,2\pi]$. Calculate the components of the 4-vectors $k'$,$q$. Define the three unit vectors $\hat{x_{q}}$,$\hat{y_{q}}$, $\hat{z_{q}}$ as 3-vectors in the laboratory rest frame. Define the boost vector 3-vector\\
    \begin{equation}\label{heitler1}
    \beta_{CM}=[\frac{\textbf{q}}{\nu+M}]_{Lab}
    \end{equation}
   
    This boosts four vectors in the $\gamma^{*}$P CM frame back to the laboratoru frame. Aplly the boost $-\beta_{CM}$ to the four-vectors q and P to obtain their versions in the $\gamma^{*}P$CM frame.
    
    \item Generate the $\pi\pi$ invariant mass squared
    \begin{equation}\label{heitler1}
    M_{1,2}^{2} \in [4m_{\pi}^{2}, (\sqrt[]{W^{2}}-M)^{2}]
    \end{equation}
    
    Calculate $P^{*}_{1,2}$
    
    \item Generate $t \in [t_{max},t_{min}]$ with the limits defined above. Calculate $cos\theta^{*}_{1,2}$. Check that $-1 < cos(\theta^{*}_{1,2}) <1$.
    \item
Generate $\phi_{1,2}^\ast \in[0,2\pi]$.\\
Compute the  $\gamma^\ast P$ CM frame four-vectors $P_{1,2}^\ast$ and $P'$.\\
Use the variables in the  $\gamma^\ast P$ CM frame to evaluate the three unit vectors $\hat x_{1,2}^\ast, \hat y_{1,2}^\ast, \hat z_{1,2}^\ast$, as defined in Eqs.~\ref{eq:unit_z12}-\ref{eq:unit_x12}
Define the boost 3-vector
\begin{equation}
\beta_{\sigma-\text{Rest}} = \left[ \frac{\mathbf P_{1,2}}{E_{1,2}}\right]_{\gamma^\ast P\,\text{CM}}
\end{equation}
This  will boost four-vectors from the $\sigma$-meson rest frame back to the $\gamma^\ast P\,\text{CM}$ frame.
\item
Generate $\cos\theta_{\sigma-\text{Rest}} \in [-1,1]$.
\item
Generate $\phi_{\sigma-\text{Rest}} \in [0,2\pi]$.\\
Compute the $\sigma\text{-Rest}$ frame four-vectors $p_1$ and $p_2$ of the two pions.
\item
Boost the four-vectors $p_1$, $p_2$ back to the  $\gamma^\ast P\,\text{CM}$ frame
\item
Boost the four-vectors $p_1$,  $p_2$, $P'$ back to the laboratory frame.
    

