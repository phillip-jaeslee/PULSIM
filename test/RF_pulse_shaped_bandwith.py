import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt

## shaped pulse calculator

"""
parameters 
M0              - start vector size of magnetization
M_equilibrium   - location of initial magnetization vector
dt              - size of each step
flip            - flip angle (rad)
t_max           - maximum time of T
BW              - bandwith (kHz)
"""

Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])
BWplot = 1 # kHz
df = np.linspace(-BWplot, BWplot, num=100)

flip = np.pi / 3

# create TBW = 4 pulse shape
TBW = 4

# shaped Pulse (sinc)
N = 100
init = -N/2
final = N/2 - 1
IN = np.arange(init, final+1, 1) / N
RF_shape = np.hamming(N).T * np.sinc(IN * TBW)

# Trf = 4 ms pulse, TBW = 4, so BWrf = TBW/Trf = 1 kHz
Trf = 4 # ms
t = IN * Trf
dt = Trf / N 
RF = (flip) * RF_shape/np.sum(RF_shape) / (2*np.pi*Gamma*dt)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)
print(np.shape(M))
for n in range(len(t)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma])
    

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')

axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, np.sqrt(M[0,:]**2+M[1,:]**2), label="|Mxy|")
axs[1].set(xlabel='frequency (kHz)', ylabel='flip')
axs[1].legend()

plt.show()



"""

% Windowed Sinc Pulse
tmax = 8;
N = tmax/dt;
t = [-N/2:N/2-1]*dt;
RF =  hamming(N)' .* sinc(t);
RF = (flip*pi/180)* RF/sum(RF) /(2*pi*gammabar*dt);


M = repmat(M_equilibrium, [1, length(df)]);
for n = 1:length(t)
    for f = 1:length(df)
        disp(bloch_rotate( M(:,f), dt, [real(RF(n)),imag(RF(n)),df(f)/gammabar]));
    end
end


subplot(211)
plot(t,RF)
xlabel('time (ms)'), ylabel('RF (mT)')
subplot(212)
plot(df,sqrt(M(1,:).^2 + M(2,:).^2), df, M(3,:))
title('Frequency profile')
xlabel('frequency (kHz)'), legend('|M_{XY}|', 'M_Z')%, ylabel('flip')
"""