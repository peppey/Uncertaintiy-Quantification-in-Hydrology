A1 = 1
w1 = 2*pi/7/24
K = 10
b = 10
Sy = 0.1
D = K*b/Sy
A2 = 0.3
w2 = pi/24
distances = [1 10 100]

x = distances(1)
solutions = []

for i = 1:1:721 % loop through timesteps
    composed_solution = solution(x, i, A1, w1,D, 0) + solution(x, i, A2, w2, D, 0)
    solutions(i) = composed_solution
end

f = figure(1)
f.Position = [0,0,1200,800]
cwt(solutions, 'amor', hours(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')



x = distances(2)
solutions = []

for i = 1:1:721 % loop through timesteps
    composed_solution = solution(x, i, A1, w1,D, 0) + solution(x, i, A2, w2, D, 0)
    solutions(i) = composed_solution
end

f = figure(2)
f.Position = [0,0,1200,800]
cwt(solutions, 'amor', hours(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')


x = distances(3)
solutions = []

for i = 1:1:721 % loop through timesteps
    composed_solution = solution(x, i, A1, w1,D, 0) + solution(x, i, A2, w2, D, 0)
    solutions(i) = composed_solution
end

f = figure(3)
f.Position = [0,0,1200,800]
cwt(solutions, 'amor', hours(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')




function res = solution(x, t, A, omega, D, phi)
    sinus = sin(-x*sqrt(omega/(2*D))+ omega*t + phi)
    res = A*exp(-x*sqrt(omega/(2*D)))*sinus
end

