import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt
from pulse import shaped_pulse, hard_pulse, import_shaped_pulse


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )



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
BW = 6 # kHz
df = np.linspace(-BW/2, BW/2, num=1000)
N = 1000

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)


file_path = 'wave/5lobe_sinc_1000'


# shaped Pulse (sine)
print(f'first pulse "{file_path}" running...')
M, df, RF, t_max =import_shaped_pulse(M, np.pi/2, "x", 0.6, file_path, BW, Gamma)


t = np.arange(0, N, 1) * t_max / N

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
df = df * 1000
axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, np.sqrt(M[0,:]**2+M[1,:]**2), label="|Mxy|")
axs[1].set(xlabel='frequency (Hz)', ylabel='flip')
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