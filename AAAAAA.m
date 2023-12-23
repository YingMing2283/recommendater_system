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

product_score_col = data(:, "product_score"); 
ps_min = min(product_score_col);
ps_max = max(product_score_col);
scaled_ps = (product_score_col - ps_min) ./ (ps_max - ps_min);
disp("Scaled Product Score:");
disp(scaled_ps);

comp_1_col = data(:, "comp_1"); 
comp_1_min = min(comp_1_col);
comp_1_max = max(comp_1_col);
scaled_comp_1 = (comp_1_col - comp_1_min) ./ (comp_1_max - comp_1_min);
disp("Scaled comp_1:");
disp(scaled_comp_1);

ps1_col = data(:, "ps1"); 
ps1_min = min(ps1_col);
ps1_max = max(ps1_col);
scaled_ps1 = (ps1_col - ps1_min) ./ (ps1_max - ps1_min);
disp("Scaled ps1:");
disp(scaled_ps1);

fp1_col = data(:, "fp1"); 
fp1_min = min(fp1_col);
fp1_max = max(fp1_col);
scaled_fp1 = (fp1_col - fp1_min) ./ (fp1_max - fp1_min);
disp("Scaled fp1:");
disp(scaled_fp1);

comp_2_col = data(:, "comp_2"); 
comp_2_min = min(comp_2_col);
comp_2_max = max(comp_2_col);
scaled_comp_2 = (comp_2_col - comp_2_min) ./ (comp_2_max - comp_2_min);
disp("Scaled comp_2:");
disp(scaled_comp_2);

ps2_col = data(:, "ps2"); 
ps2_min = min(ps2_col);
ps2_max = max(ps2_col);
scaled_ps2 = (ps2_col - ps2_min) ./ (ps2_max - ps2_min);
disp("Scaled ps2:");
disp(scaled_ps2);

fp2_col = data(:, "fp2"); 
fp2_min = min(fp2_col);
fp2_max = max(fp2_col);
scaled_fp2 = (fp2_col - fp2_min) ./ (fp2_max - fp2_min);
disp("Scaled fp2:");
disp(scaled_fp2);

comp_3_col = data(:, "comp_3"); 
comp_3_min = min(comp_3_col);
comp_3_max = max(comp_3_col);
scaled_comp_3 = (comp_3_col - comp_3_min) ./ (comp_3_max - comp_3_min);
disp("Scaled comp_3:");
disp(scaled_comp_3);

ps3_col = data(:, "ps3"); 
ps3_min = min(ps3_col);
ps3_max = max(ps3_col);
scaled_ps3 = (ps3_col - ps3_min) ./ (ps3_max - ps3_min);
disp("Scaled ps3:");
disp(scaled_ps3);

fp3_col = data(:, "fp3"); 
fp3_min = min(fp3_col);
fp3_max = max(fp3_col);
scaled_fp3 = (fp3_col - fp3_min) ./ (fp3_max - fp3_min);
disp("Scaled fp3:");
disp(scaled_fp3);

lag_price_col = data(:, "lag_price"); 
lag_price_min = min(lag_price_col);
lag_price_max = max(lag_price_col);
scaled_lag_price = (lag_price_col - lag_price_min) ./ (lag_price_max - lag_price_min);
disp("Scaled lag_price:");
disp(scaled_lag_price);

qty_col = 4;
unit_cost_col = 7;
freight_price_col = 6;
comp_1_price_col = 21;
comp_2_price_col = 24;
comp_3_price_col = 27;
product_rating_col = 12;
min_rating_col = ps_min;
comp_1_rating_col = 22;
comp_2_rating_col = 25;
comp_3_rating_col = 28;
comp_1_freight_price_col = 23;
comp_2_freight_price_col = 26;
comp_3_freight_price_col = 29;
lag_price_col = 30;

% Define the coefficients of the objective function
f = [0; -(data{:, unit_cost_col} - data{:, freight_price_col}); 0];

% Define the inequality constraints matrix A and vector b
n = size(data, 1);
A = [zeros(n, 1), -eye(n), zeros(n, 1);
     zeros(n, 1), zeros(n, n), -eye(n);
     -eye(n), zeros(n, n), zeros(n, 1);
     zeros(n, 1), zeros(n, n), -eye(n)];
b = [zeros(3*n, 1); -(data{:, comp_1_price_col} - data{:, comp_1_freight_price_col});
     -(data{:, comp_2_price_col} - data{:, comp_2_freight_price_col});
     -(data{:, comp_3_price_col} - data{:, comp_3_freight_price_col})];

% Define the lower bounds and upper bounds for the decision variables
lb = [0; -Inf; -Inf];
ub = [Inf; Inf; Inf];

% Define the equality constraints matrix Aeq and vector beq
Aeq = [data{:, lag_price_col} - data{:, unit_cost_col}, data{:, qty_col}];
beq = data{:, qty_col} - data{:, lag_price_col} .* data{:, comp_1_price_col}';

% Solve the linear programming problem
options = optimoptions('linprog', 'Display', 'iter');
[x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, options);

% Check the optimization result
if exitflag == 1
    disp("Optimization successful!");
    disp("Optimal solution:");
    disp(x);
    disp("Optimal profit:");
    disp(-fval);
else
    disp("Optimization failed.");
end




