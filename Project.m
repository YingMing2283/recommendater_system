data = csvread("C:/Users/user/Desktop/retail_price.csv");

% Split the dataset into relevant variables
product_id = data(:, 1);
product_category_name = data(:, 2);
month_year = data(:, 3);
qty = data(:, 4);
total_price = data(:, 5);
freight_price = data(:, 6);
unit_price = data(:, 7);
product_name_length = data(:, 8);
product_description_length = data(:, 9);
product_photos_qty = data(:, 10);
product_weight_g = data(:, 11);
product_score = data(:, 12);
customers = data(:, 13);
weekday = data(:, 14);
weekend = data(:, 15);
holiday = data(:, 16);
month = data(:, 17);
year = data(:, 18);
s = data(:, 19);
volume = data(:, 20);
comp_1 = data(:, 21);
ps1 = data(:, 22);
fp1 = data(:, 23);
comp_2 = data(:, 24);
ps2 = data(:, 25);
fp2 = data(:, 26);
comp_3 = data(:, 27);
ps3 = data(:, 28);
fp3 = data(:, 29);
lag_price = data(:, 30);

% Handle missing values
% Assuming missing values are represented by NaN
% Replace missing values with the mean value of the corresponding variable
missing_indices = isnan(qty);
qty(missing_indices) = mean(qty, 'omitnan');

