% (a)
% Clear parameter values, reset.
clear; clc;
rng(1);

% Calibrartion of parameters.
beta=0.96;
gamma=1.3;
r=0.04;
rho=0.9;
sigma=0.04;
n=5; % Number of states
m=3; %max +- 3 std. devs for normal distribution input in Tauchen function.
mu=0; %unconditional mean of process.

% Define Y (5x1) space using Tauchen method ie Z. P is Markov Chain (5x5)
% ie Zprob for good estimation of normal distribution.
[Y,P]=Tauchen(n,mu,rho,sigma,m);
%Y=Y(:)'; % Transpose entire column vector of income states to row vector.

% Define asset a space.
amin=min(Y(1))/r;
amax=100; % Large enough upperlimit for assets (arbitrary).
a=linspace(amin,amax,300); % column --> transpose to row
%a=a(:)'; % row
% Equidistant distribution of a with lowerbound amin and upperbound amax of
% 300 points.
% Transpose to convert row vector to column vector of 300x1.

% Inicialization guess of V0 as more rows than columns. Guess same thing
% for all.
%a_n=300; % Number of assets % Removed for simplicity.
%y_n=5; % Number of income y states % Removed for simplicity.
V0 = zeros(300,5); % Grid space.

% Inicialize V1.
V1=zeros(300,5);

% Inicialize next period assets a'.
a_function=zeros(300,5);

% Inicialize consumption c function.
c_function=zeros(300,5,300); % c function of a by y states x by a_function a'

% Construct consumption c to search for asset a which maximizes the
% objective funcion across asset space of 1 to 300. c = (1+r)a + e^y - a'
% repmat to construct consumption matrix for entire asset a, income y
% state, future asset a' combintations.
% Brute force--> plug in c and search over a range to find max a.
% a = (300,1)--> repmat --> (300,1*5)= (300,5)
% Y' = (1,5)--> repmat --> (1*300,5)= (300,5)
% a(a_range) = (1,1)--> repmat --> (1*300,1*5)= (300,5)
for  a_range=1:300
    c(:,:, a_range)=(1+r)*repmat(a',1,5)+exp(repmat(Y',300,1))-repmat(a(a_range),300,5);
end
% Note: Number of income states = 5 on the grid.

% Set inicial tollerance. Note: 10^-9 in instructions. Start small and converge.
e=1e-9;
% Counting condition to prevent an infinity run. Sufficiently large enough 
% maximum number of itterations.
max_iter=10000;


% Define CRRA utility function for gamma = 1 and =/ 1 using if function.
% Note: Dr. Ciarliero mentioned must exclude negaive c choices using -inf
% function since negative c will not work with the VFI.

% First, for all negative consumption values, consunmption =0.
c=max(c,0); % For values c<0, c=0, otherwise c=c

% Then use 'if' function, including how to deal with c=0 using -inf.
if gamma==1
    u=log(c);
else
    u=(c.^(1-gamma))/(1-gamma);
    u(c==0)=-inf; % Utility function for consumption values of zero should be negative infinity.
end


% Value Function Itteration Loop. Note multiple by P markov chain to 
% inlcude expectations. Recall nested if statements from lecture.
% Iniciallizing diffV.
current_iter=0;
norm_V=0.0005;

while (norm_V>e) && (current_iter<max_iter) % && --> AND
   % abs((V1-V0))>e
   V_guess=zeros(300,300); % Iniciallization, current a* choice future a'.
   for y_state=1:5
       for a_range = 1:300 % Note: V_guess here is (y,a) --> will transpose later s.t. (a,y) 
           % Note: P = 1x5, repmatV0=300x5
           %V0*P'=300x5 * 5x1
           V_guess(:,a_range) = u(:,y_state,a_range) + beta*repmat(V0(a_range,:),300,1)*P(y_state,:)';
           %V_guess = 300x1 + 300x5*5x1= 300x1 + 300x1= 300x1
       end
       [V1(:,y_state),a_function(:,y_state)]=max(V_guess');
   end

   norm_V=max(max(abs(abs(V1-V0)))); % V1 is matrix, take double max and abs s.t. take max of 
   % each column, return row of maximum values, then takes maximum value of that row to return
   % scalar of maximum value.
   V0=V1; % Set resulting value V_guess' to V1 and set V1 to origional 
   % guess V0.
   current_iter=current_iter+1;
end

% Compute c posicy function given previous VFI.
% Recall: c(:,:, a_range)=(1+r)*repmat(a,1,5)+exp(repmat(Y,300,1))-repmat(a(a_range),300,5);
c=(1+r)*repmat(a',1,5)+exp(repmat(Y',300,1))- a(a_function); % c=300x5,
% Note here a_function NOT a_range because we wnat a 300x5 matrix not scalar
% For each element in a_function, go to position in a and return that value of asset itself.

% (b)
% Plot V(a,y) for all y on one axes.
figure(1); hold on; grid on;
plot(a, V1, 'LineWidth', 1.8);    % V1 is a_n x Y_n
xlabel('Assets  a');
ylabel('Value  V(a,y)');
title('Converged Value Function by Income State');
hold off;

% (c) 1000 Simulations.
% epsilon=randn(1000);

% Simulate Y using dtmc(P) which creates a discrete-time, finite-state,
% time-homogeneous Markov chain object. Markov chain is useful here (as
% states in class) because it stands for good estimation of normal
% distribution. Used randn to have normal distribution.
% X = simulate(mc,numSteps) returns data X on random walks of length 
% numSteps (1000 in this case) through sequences of states in the 
% discrete-time Markov chain.
P_object=dtmc(P);

T=1000;
drop=500;

simulate_Y=simulate(P_object,T); % need Y to compute c and a'
a_inicial=1; % inicializing

for t=1:T
    simulate_c(t)=(1+r)*a(a_inicial(t)) + exp(Y(simulate_Y(t))) - a(a_function(a_inicial(t), simulate_Y(t)));
    a_inicial(t+1)=a_function(a_inicial(t), simulate_Y(t));
    simulate_a(t+1)=a(a_inicial(t+1)); % sim_a is next period assets
end

figure(2);
tiledlayout(3,1, "TileSpacing","compact", "Padding","compact");

nexttile;
plot(drop+1:T, simulate_Y(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Income');

nexttile;
plot(drop+1:T, simulate_a(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Next Period Assets');

nexttile;
plot(drop+1:T, simulate_c(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Consumption'); 

% (d)
% std calculates the s.d. of ur simulated c only for the last 500 simulations.
std_c=std(simulate_c(drop+1:T))
% Output: s.d. of c = 0.0904

%  (a) The borrowing constraint were zero.
% A more restrictive borrowing restraint (ie. no borrowing) would make it more
% diffucult for households to smooth consumption. As such, consumption
% would experience more varaition relative to before as it tracks income more closely. 
% Standard deviation would therefore be quantitatively higher.

% (b) The relative risk aversion parameter doubled.
% A greater RRA parameter would increase households desire to smooth
% consumption as they are more sensitive to consumption changes. As such, 
% standard deviation is expected to be quantitatively smaller.

%  (c) The natural rate of interest doubled.
% A higher natural rate of return is expected to make it easier for houesholds to
% smooth consumption as they recive a larger return on investment. 
% Standard deviation is thus expected to be lower.

% (d) Income volatility doubled.
% Higher income volitility makes it more challenging for households to
% maintain a steady level of consumption, regardless of buffer-savings. As

% such, standard deviation of consumption is expected to rise.
