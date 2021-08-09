function MAE = computeMAE_func(f_func, x_arr, y_real_arr)
% compute the mean absolute error (MAE) for the given function

FUN = matlabFunction(f_func); % This creates a function handle

y_func_arr = feval(FUN, x_arr);

MAE = median(abs(y_real_arr - y_func_arr)); % mean median

end