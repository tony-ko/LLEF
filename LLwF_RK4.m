%% Lugiato-Lefever equation with delay and modulation
% Here we numerically solve Lugiato-Lefever equation with a delay following
% the formulation in M. Tlidi, et. al., Chaos 27, 114312 (2017)
% the equation is solved by the split-step Fourier method.
%
% The equation:
%
% D[E(t,xi),t] = i*b*D[E(t,xi),xi,xi]
%                -(1+i*th)*E(t,xi)+i*abs(E(t,xi))^2*E(t,xi)+E_in(t)
%                +et*exp(i*ph)*E(t-tau,xi)
%
% et can be zero (regular LLE), constant (LLE w/feedback) or a function.
%
% xi is fast time inside the cavity (subject to a periodic boundary
% condition),
%
% t is slow evolution time
%
% The linear part is pump, detuning, losses and dispersion:
%
% L = i*b*D[E(t,xi),xi,xi]-(1+i*th)*E(t,xi)+E_in(t)
%
% The nonlinear part is Kerr frequency shift and feedback:
%
% N = i*abs(E(t,xi))^2*E(t,xi) + et*exp(i*ph)*E(t-tau,xi)
%
% The linear part is done by FFT,
% and nonlinear is by 4th order Runge-Kutta method

clear all
%% here we define the space and time grids
% number of modes in Fourier space for the fast intracavity time
nF = 512;
% slow time step duration
dt = 0.25;
% RK substeps
nRK = 1;

% integration ranges
% fast time
S = 50;
% slow time
endtime = 20000;

% the further parameters are calculated
% number of steps to take in slow time
M = round(endtime/dt);
% RK substep
dtRK = dt/nRK;

% fast time step
h = S/nF;
% spectral mode indices
n = (-nF/2:1:nF/2-1)';
% fast time grid points
x = n*h;
% fast time wavenumbers
k = 2*pi*n/S;

%% here we solve the model by SSFM
% parameters for the equation
b = 1;
th = 3;
et = 0;
ph = pi;
tau = 100;
E_in = 2;

% initial condition u(t;t<=0,xi), and also this is the history function
% it can be just some sech pulse
u0 = sech(x/4);

% or we can import the precomputed cavity soliton solution
% re_u0=importdata('re_u0_1024.csv');
% im_u0=importdata('im_u0_1024.csv');
% u0 = re_u0 + 1i*im_u0;
clear re_u0 im_u0

% the matrix with all calc results
U = zeros(M,nF);

% the number of previous iteration corresponding to the delay term
nhist = ceil(tau/dt);

fE_in=repmat(E_in,nF,1);
fE_in=fftshift(fft(fE_in));
u = u0;

hist_len = nhist*nRK;

init_hist = repmat({u0},hist_len,1);

% our initial history for RK method
qY1 = init_hist;
qY1_head = 1;
qY1_tail = hist_len;

qY2 = init_hist;
qY2_head = 1;
qY2_tail = hist_len;

qY3 = init_hist;
qY3_head = 1;
qY3_tail = hist_len;

qY4 = init_hist;
qY4_head = 1;
qY4_tail = hist_len;
% clear init_hist
tic
% the loop in slow time
for m = 1:1:M
    % linear part regain only pump and losses terms
    const_term = fE_in;
    prop = -1i*b*k.^2 - (1 + 1i*th);
    exp_prop = exp(prop*dt/2);
    % this can seem incoherent, but this is how you introduce a constant
    % term in SSFM
    const_add = const_term.*(exp_prop - 1)./prop;
    
    % we propagate the solution half-step in Fourier domain
    v = fftshift(fft(u));
    v = const_add + exp_prop.*v;
    v = ifft(fftshift(v));
    
    % gradual feedback increase
    et = 0.02*fix(m/round(3000/dt));
    
    % we propagate the nonlinear part by RK substeps
    % we use queue structure to keep intermediate steps
    
    
    for l=1:1:nRK
        % boresome operators over queues
        % q 1
        Y1T = qY1{qY1_head};
        qY1_head = qY1_head + 1;
        if qY1_head > hist_len
            qY1_head = 1;
        end
        
        % q 2
        Y2T = qY2{qY2_head};
        qY2_head = qY2_head + 1;
        if qY2_head > hist_len
            qY2_head = 1;
        end
        
        % q 3
        Y3T = qY3{qY3_head};
        qY3_head = qY3_head + 1;
        if qY3_head > hist_len
            qY3_head = 1;
        end
        
        % q 4
        Y4T = qY4{qY4_head};
        qY4_head = qY4_head + 1;
        if qY4_head > hist_len
            qY4_head = 1;
        end
        
        % yay, some RK calculations!
        Y2 = v + 0.5.*dtRK.*LLwF_LLN(v,Y1T,et,ph);
        Y3 = v + 0.5.*dtRK.*LLwF_LLN(Y2,Y2T,et,ph);
        Y4 = v + dtRK.*LLwF_LLN(Y3,Y3T,et,ph);
        
        % the same boredom for the queues tails
        qY1_tail = mod(qY1_tail, hist_len) + 1;
        qY1{qY1_tail} = v;
 
        qY2_tail = mod(qY2_tail, hist_len) + 1;
        qY2{qY2_tail} = Y2;
        
        qY3_tail = mod(qY3_tail, hist_len) + 1;
        qY3{qY3_tail} = Y3;
        
        qY4_tail = mod(qY4_tail, hist_len) + 1;
        qY4{qY4_tail} = Y4;
        
        f1 = LLwF_LLN(v,Y1T,et,ph);
        f2 = LLwF_LLN(Y2,Y2T,et,ph);
        f3 = LLwF_LLN(Y3,Y3T,et,ph);
        f4 = LLwF_LLN(Y4,Y4T,et,ph);
        v = v +dtRK.*(f1./6 + f2./3 + f3./3 + f4./6);
    end
    
    % then propagate it half-step further
    v = fftshift(fft(v));
    v = const_add + exp_prop.*v;
    u = ifft(fftshift(v));
       
    % add to the field storage
    U(m,:) = u';
end
clear qY1 qY2 qY3 qY4
clear qY1_tail qY2_tail qY3_tail qY4_tail
clear qY1_head qY2_head qY3_head qY4_head
clear f1 f2 f3 f4 exp_prop prop const_add const_term
clear Y1 Y1T Y2 Y2T Y3 Y3T Y4 Y4T u0
toc
%% export the last one to use it further
u0 = U(end,:)';
dlmwrite('re_u0_1024.csv',real(u0),'delimiter',',','precision',5);
dlmwrite('im_u0_1024.csv',imag(u0),'delimiter',',','precision',5);
%% projected surface plot (or density plot)
figure(2);clf
xscale = 50;
yscale = 1;
imagesc(abs(U(1:xscale:end,1:yscale:end)).^2)
set(gca, 'xtick', 1:nF/(10*yscale):nF)
set(gca, 'xticklabels', -S/2:S/10:S/2)
set(gca, 'ytick', 1:M/(20*xscale):M/xscale)
set(gca, 'yticklabels', 0:endtime/20:endtime)
h = colorbar;
ylabel(h, '|E|^2')
xlabel('\xi')
ylabel('y')
axis tight
clear xscale yscale h
%% stacked plot
figure(3);clf
xscale = 3000;
yscale = 1;
h = waterfall(abs(U(1:xscale:end,1:yscale:end)).^2);
colormap([1 1 1]);
color = repmat([0 0 0],length(h.FaceVertexCData),1);
set(h, 'FaceVertexCData', color)
set(h, 'FaceColor', 'flat')
set(gca,'Color','k')
set(h, 'EdgeColor', [1 1 1])
set(gca, 'xtick', 1:nF/(10*yscale):nF)
set(gca, 'xticklabels', -S/2:S/10:S/2)
set(gca, 'ytick', 1:M/(10*xscale):M/xscale)
set(gca, 'yticklabels', 0:endtime/10:endtime)
view([0 75])
axis tight
xlabel('\xi')
ylabel('t')
zlabel('|E|^2')
clear h xscale yscale color
%% surface plot
figure(1);clf
xscale = 100;
yscale = 2;
s = surf(abs(U(1:xscale:end,1:yscale:end)).^2);
s.EdgeColor = 'none';
set(gca, 'xtick', 1:nF/(10*yscale):nF)
set(gca, 'xticklabels', -S/2:S/10:S/2)
set(gca, 'ytick', 1:M/(10*xscale):M/xscale)
set(gca, 'yticklabels', 0:endtime/10:endtime)
axis tight
xlabel('\xi')
ylabel('t')
zlabel('|E|^2')
clear xscale yscale
view(3)
