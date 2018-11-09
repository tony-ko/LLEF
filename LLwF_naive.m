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
% The nonlinear part is traditional for NLSE:
% 
% N = i*abs(E(t,xi))^2*E(t,xi)
%
% The linear part is all the rest:
%
% L = i*b*D[E(t,xi),xi,xi]-(1+i*th)*E(t,xi)+E_in(t)
%                +et*exp(i*ph)*E(t-tau,xi)
%
% the naive approach assumes that the delayed term is constant
% for a small step, which is not necessary true.
%
% The linear part is solved by FFT, nonlinear part is solved
% by propagating the exact solutions.

clear all;
%% here we define the space and time grids
% number of modes in Fourier space for the fast intracavity time
nF = 1024;
% slow time step duration
dt = 0.1;

% integration ranges
% fast time (cavity round-trip)
S = 50;
% slow time
endtime = 10000;

% the further parameters are calculated
% number of time steps to take
M = round(endtime/dt);

% fast time step value
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
% re_u0=importdata('re_u0_512.csv');
% im_u0=importdata('im_u0_512.csv');
% u0 = re_u0 + 1i*im_u0;
clear re_u0 im_u0

% the matrix with all the calculated results
U = zeros(M,nF);

% the number of previous iterations corresponding to the delay term
nhist = ceil(tau/dt);

% FFT of pump field to be accounted during FFT step
fE_in=repmat(E_in,nF,1);
fE_in=fftshift(fft(fE_in));

% our initial field
u = u0;

% for the performance estimation
tic

% the loop in slow time
for m = 1:1:M
    % define delayed term
    if m - nhist < 1
        uhist = u0;
    else
        uhist = U(m - nhist,:)';
    end
    % gradual feedback increase if necessary
    % et = 0.02*fix(m/round(3000/dt));
    
    % FFT of delayed field value
    fuhist=fftshift(fft(et*exp(1i*ph)*uhist));
    const_term = fuhist + fE_in;
    prop = -1i*b*k.^2 - (1 + 1i*th);
    exp_prop = exp(prop*dt/2);
    
    % this can seem incoherent, but this is how you introduce a constant
    % term in SSFM
    const_add = const_term.*(exp_prop - 1)./prop;
    
    % we propagate the solution half-step in Fourier domain
    c = fftshift(fft(u));
    c = const_add + exp_prop.*c;
    u = ifft(fftshift(c));    
    
    % apply nonlinear part for the full step
    u = exp(1i*dt*(abs(u).^2)).*u;
    
    % then propagate it half-step further
    c = fftshift(fft(u));
    c = const_add + exp_prop.*c;
    u = ifft(fftshift(c));   
    
    % add to the field storage (eats memory and time,
    %  you can do it less often)
    U(m,:) = u';    
    
end
clear c exp_prop prop const_add const_term fuhist
% print time spent
toc
%% export the last one to use it further
u0 = U(end,:)';
dlmwrite('re_u0_1024.csv',real(u0),'delimiter',',','precision',5);
dlmwrite('im_u0_1024.csv',imag(u0),'delimiter',',','precision',5);
%% projected surface plot (or density plot)
figure(3);clf
xscale = 100;
yscale = 1;
imagesc(abs(U(1:xscale:end,1:yscale:end)).^2)
set(gca, 'xtick', 1:nF/(10*yscale):nF)
set(gca, 'xticklabels', -S/2:S/10:S/2)
set(gca, 'ytick', 1:M/(10*xscale):M/xscale)
set(gca, 'yticklabels', 0:endtime/10:endtime)
h = colorbar;
ylabel(h, '|E|^2')
xlabel('\xi')
ylabel('t')
axis tight
clear xscale yscale h
%% joy division plot
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
clear h xscale yscale
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