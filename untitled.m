% Read the dataset
data = readtable('C:\Users\user\Desktop\retail_price.csv');

% Extract the required columns
qty_col = data.qty;
unit_cost_col = data.unit_price;
freight_price_col = data.freight_price;
comp_1_price_col = data.comp_1;
comp_2_price_col = data.comp_2;
comp_3_price_col = data.comp_3;
product_rating_col = data.product_score;
comp_1_rating_col = data.ps1;
comp_2_rating_col = data.ps2;
comp_3_rating_col = data.ps3;
comp_1_freight_price_col = data.fp1;
comp_2_freight_price_col = data.fp2;
comp_3_freight_price_col = data.fp3;
lag_price_col = data.lag_price;

% Calculate the scaled variables
scaled_qty = (qty_col - min(qty_col)) ./ (max(qty_col) - min(qty_col));
scaled_fp = (freight_price_col - min(freight_price_col)) ./ (max(freight_price_col) - min(freight_price_col));
scaled_comp_1 = (comp_1_price_col - min(comp_1_price_col)) ./ (max(comp_1_price_col) - min(comp_1_price_col));
scaled_ps1 = (comp_1_rating_col - min(comp_1_rating_col)) ./ (max(comp_1_rating_col) - min(comp_1_rating_col));
scaled_fp1 = (comp_1_freight_price_col - min(comp_1_freight_price_col)) ./ (max(comp_1_freight_price_col) - min(comp_1_freight_price_col));
scaled_comp_2 = (comp_2_price_col - min(comp_2_price_col)) ./ (max(comp_2_price_col) - min(comp_2_price_col));
scaled_ps2 = (comp_2_rating_col - min(comp_2_rating_col)) ./ (max(comp_2_rating_col) - min(comp_2_rating_col));
scaled_fp2 = (comp_2_freight_price_col - min(comp_2_freight_price_col)) ./ (max(comp_2_freight_price_col) - min(comp_2_freight_price_col));
scaled_comp_3 = (comp_3_price_col - min(comp_3_price_col)) ./ (max(comp_3_price_col) - min(comp_3_price_col));
scaled_ps3 = (comp_3_rating_col - min(comp_3_rating_col)) ./ (max(comp_3_rating_col) - min(comp_3_rating_col));
scaled_fp3 = (comp_3_freight_price_col - min(comp_3_freight_price_col)) ./ (max(comp_3_freight_price_col) - min(comp_3_freight_price_col));
scaled_lag_price = (lag_price_col - min(lag_price_col)) ./ (max(lag_price_col) - min(lag_price_col));

% Define the objective function
objective = @(x) -sum(scaled_qty .* (unit_cost_col - x));

% Define the constraint functions
min_rating_col = min(product_rating_col); % Minimum rating column value
price_comparison_constraint = @(x) [x - scaled_comp_1; x - scaled_comp_2; x - scaled_comp_3];
product_rating_constraint = @(x) [product_rating_col - min_rating_col; product_rating_col - scaled_comp_1; product_rating_col - scaled_comp_2; product_rating_col - scaled_comp_3];
freight_charges_constraint = @(x) [x - scaled_fp - (scaled_comp_1 - scaled_fp1); x - scaled_fp - (scaled_comp_2 - scaled_fp2); x - scaled_fp - (scaled_comp_3 - scaled_fp3)];

% Define the lower and upper bounds
lb = 0; % Lower bound for variable x
ub = 1; % Upper bound for variable x

% Set optimization options for the genetic algorithm
options = optimoptions('ga', 'Display', 'off');

% Define the nonlinear inequality constraints
nonlcon = @(x) [
    quantity_constraint(x);
    minimum_profit_constraint(x);
    price_comparison_constraint(x);
    product_rating_constraint(x);
    freight_charges_constraint(x)
];

% Solve the optimization problem
[x, profit] = ga(objective, 1, [], [], [], [], lb, ub, nonlcon, options);

% Display the results
disp(['Optimal Profit: ', num2str(-profit)]);
disp(['Optimal x: ', num2str(x)]);
