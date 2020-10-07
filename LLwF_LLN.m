%% Lugiato-Lefever nonlinear RHS
function dy = LLwF_LLN(y,yT,et,ph)
    dy = 1i.*abs(y).^2.*y + et.*exp(-1i.*ph).*yT;
end