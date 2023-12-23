data = readtable('C:\Users\user\Desktop\retail_price.csv');

qty_col = data(:, "qty");
qty_min = min(qty_col);
qty_max = max(qty_col);
scaled_qty = (qty_col - qty_min) ./ (qty_max - qty_min);
disp("Scaled Quantity Sold:");
disp(scaled_qty);

freight_price_col = data(:, "freight_price"); 
fp_min = min(freight_price_col);
fp_max = max(freight_price_col);
scaled_fp = (freight_price_col - fp_min) ./ (fp_max - fp_min);
disp("Scaled Freight Price:");
disp(scaled_fp);

f = [-41 -60 -14 -55 -29 -41 -62 -46 -85 172]

A = [1   0   0   0   0   0   0   0   0   0;
     0   1   0   0   0   0   0   0   0   0;
     0   0   1   0   0   0   0   0   0   0;
     0   0   0   1   0   0   0   0   0   0;
     0   0   0   0   1   0   0   0   0   0;
     0   0   0   0   0   1   0   0   0   0;
     0   0   0   0   0   0   1   0   0   0;
     0   0   0   0   0   0   0   1   0   0;
     0   0   0   0   0   0   0   0   1   0;
     0   0   0   0   0   0   0   0   0   1;]
 

%B = [125 250 55 105 155 440 145 40 280 80 ] 
B = [125;250; 55; 105; 155; 440; 145; 40 ;280; 80] 

A_eq = []

B_eq = []

lb = [0 0 0 0 0 0 0 0 0 0 ]

ub = []

options = optimoptions('linprog', 'Algorithm','interior-point', 'Display', 'iter', 'Maxiterations', 2000, 'OptimalityTolerance', 1e-9, 'ConstraintTolerance',1e6)
[x, fval, exitflag, output] = linprog(f,A,B,A_eq,B_eq,lb,ub, options)
%[X, Z] = linprog (f,A,B,A_eq,B_eq,lb,ub)

% Run the linprog function
[x, fval, exitflag, output] = linprog(f, A, B, A_eq, B_eq, lb, ub, options);

% Check the exitflag
if exitflag == 1
    disp('Optimization successful!');
else
    disp('Optimization did not converge.');
end

% Display the optimal solution vector
disp('Optimal solution:');
disp(x);

% Display the optimal objective function value
disp('Optimal objective value:');
disp(fval);

% Display the output information
disp('Output information:');
disp(output);

% Example code for further analysis and interpretation

% Extract the allocated amounts from the solution vector x
allocatedAmounts = x(1:10);

% Display the allocated amounts
disp('Allocated amounts:');
disp(allocatedAmounts);

% Total allocated amount
totalAllocated = sum(allocatedAmounts);
disp('Total allocated amount:');
disp(totalAllocated);

% Calculate the percentage of allocation for each resource
allocationPercentage = (allocatedAmounts ./ totalAllocated) * 100;

% Display the allocation percentage for each resource
disp('Allocation percentage:');
disp(allocationPercentage);

% Identify any constraints that are close to being violated
constraintViolations = A * allocatedAmounts - B;

% Display the constraint violations
disp('Constraint violations:');
disp(constraintViolations);

% Check if any constraints are close to being violated
if any(constraintViolations > 0)
    disp('Some constraints are close to being violated.');
else
    disp('All constraints are satisfied.');
end

% Perform sensitivity analysis by adjusting the objective function coefficients
disp('Sensitivity Analysis:');
new_f = [1.1*f(1:9), 0.9*f(10)];
[x_sensitivity, fval_sensitivity] = linprog(new_f, A, B, A_eq, B_eq, lb, ub, options);
disp('New optimal solution (sensitivity analysis):');
disp(x_sensitivity);
disp('New optimal objective value (sensitivity analysis):');
disp(fval_sensitivity);

% Validate the obtained solution
constraintViolations = A * x - B;
if all(constraintViolations <= 0)
    disp('Solution is valid.');
else
    disp('Solution is invalid. Some constraints are violated.');
end

% Scenario Analysis: Modify constraints and re-run optimization
disp('Scenario Analysis:');
% Example: Increase resource availability
new_B = B + 50;
[x_scenario, fval_scenario] = linprog(f, A, new_B, A_eq, B_eq, lb, ub, options);
disp('New optimal solution (scenario analysis):');
disp(x_scenario);
disp('New optimal objective value (scenario analysis):');
disp(fval_scenario);

% Visualization
disp('Visualization:');
% Example: Bar chart of allocated amounts
figure;
bar(allocatedAmounts);
xlabel('Resource');
ylabel('Allocated Amount');
title('Allocated Amounts for Resources');

% Incorporate additional constraints
disp('Incorporating Additional Constraints:');
% Example: Additional equality constraint
new_A_eq = [A_eq; f];
new_B_eq = [B_eq; fval];
[x_additional, fval_additional] = linprog(f, A, B, new_A_eq, new_B_eq, lb, ub, options);
disp('New optimal solution (with additional constraints):');
disp(x_additional);
disp('New optimal objective value (with additional constraints):');
disp(fval_additional);

% Model Refinement
disp('Model Refinement:');
% Example: Adjust objective function and re-run optimization
new_f = [-f(1:9), f(10)];
[x_refined, fval_refined] = linprog(new_f, A, B, A_eq, B_eq, lb, ub, options);
disp('New optimal solution (refined model):');
disp(x_refined);
disp('New optimal objective value (refined model):');
disp(fval_refined);
