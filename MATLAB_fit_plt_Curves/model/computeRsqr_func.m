function Rsqr = computeRsqr_func(f_func, x_arr, y_real_arr)
% compute the R^2 for the given function

FUN = matlabFunction(f_func); % This creates a function handle

y_func_arr = feval(FUN, x_arr);

correlation_coeff = corr2(y_real_arr,y_func_arr);
Rsqr = power(correlation_coeff,2);

end