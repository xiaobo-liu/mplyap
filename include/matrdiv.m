function C = matrdiv(A, B, prec)
% matrix right division in double/single/half precisions (half-precision 
% div is done in double and then converted to simulated half).

if isequal(prec,'double') % use double precision
        C = double(A)/double(B);
elseif isequal(prec,'single') % use single precision
        C = double(single(A)/single(B));
else % use half precision
        C = double(chop(A)/chop(B));
end
end
