%% example_usage.m
% Load pre-trained model and predict scattering far-field patterns.
% Author: Zehan Liu (liuzehan@bjtu.edu.cn)
clc; close all;

%% Load model
load('model.mat', 'model');

thetaGrid = model.thetaGrid;
phiGrid   = model.phiGrid;
nTheta    = numel(thetaGrid);
nPhi      = numel(phiGrid);
cfg       = model.config;

rTarget = round((nTheta+1)/2);
cTarget = round((nPhi+1)/2);
thetaCenter = thetaGrid(rTarget);
phiCenter   = phiGrid(cTarget);
[phiMesh, thetaMesh] = meshgrid(phiGrid, thetaGrid);
thetaRel = thetaMesh - thetaCenter;
phiRel   = mod(phiMesh - phiCenter + 180, 360) - 180;

%% Single prediction
U = 3;  freqGHz = 1.5;  thetaIncDeg = 30;  pol = 'TE';

Emap = predict_field(model, cfg, thetaRel, phiRel, U, freqGHz, thetaIncDeg, pol, 'single_lobe');

figure('Color','w');
imagesc(phiGrid, thetaGrid, Emap); axis xy; colorbar;
xlabel('\phi (°)'); ylabel('\theta (°)');
title(sprintf('U=%d  f=%.1f GHz  \\theta_i=%d°  %s', U, freqGHz, thetaIncDeg, pol));

%% Batch prediction
params = {
    3, 1.5, 30, 'TE',  'single_lobe';
    5, 2.0, 45, 'TE',  'single_lobe';
    3, 3.0, 60, 'TM',  'single_lobe';
    7, 1.0, 10, 'TM',  'phi_broad';
};

figure('Color','w');
for n = 1:size(params,1)
    E = predict_field(model, cfg, thetaRel, phiRel, ...
        params{n,1}, params{n,2}, params{n,3}, params{n,4}, params{n,5});
    subplot(2, 2, n);
    imagesc(phiGrid, thetaGrid, E); axis xy; colorbar;
    xlabel('\phi (°)'); ylabel('\theta (°)');
    title(sprintf('U=%d f=%.1f \\theta_i=%d° %s', params{n,1}, params{n,2}, params{n,3}, params{n,4}));
end

%% Compare with FEKO training data stored in model
idx = 1;
Y_true = reshape(model.Y_feko(idx,:), nTheta, nPhi);
Y_hat  = reshape(model.Y_pred(idx,:), nTheta, nPhi);

figure('Color','w');
subplot(1,3,1);
imagesc(phiGrid, thetaGrid, Y_true); axis xy; colorbar;
xlabel('\phi (°)'); ylabel('\theta (°)'); title('FEKO |E|');

subplot(1,3,2);
imagesc(phiGrid, thetaGrid, Y_hat); axis xy; colorbar;
xlabel('\phi (°)'); ylabel('\theta (°)'); title('Model');

subplot(1,3,3);
imagesc(phiGrid, thetaGrid, Y_true - Y_hat); axis xy; colorbar;
xlabel('\phi (°)'); ylabel('\theta (°)'); title('Error');

c = corr(Y_true(:), Y_hat(:));
r = sqrt(mean((Y_true(:) - Y_hat(:)).^2));
fprintf('Sample %d — Corr: %.4f, RMSE: %.6f\n', idx, c, r);

%% Export to CSV
[phiOut, thetaOut] = meshgrid(phiGrid, thetaGrid);
writematrix([thetaOut(:), phiOut(:), Emap(:)], 'predicted_field.csv');
fprintf('Saved predicted_field.csv (%d rows)\n', nTheta * nPhi);

%% =================== Helper functions ===================
function Emap = predict_field(model, cfg, thetaRel, phiRel, U, freqGHz, thetaIncDeg, pol, sampleType)
    X = build_features_local(U, freqGHz, thetaIncDeg, pol);

    dsP = zeros(1, 7);
    for j = 1:7
        dsP(j) = predict_local(model.dsModels{j}, X);
    end
    dsP(1:3) = max(dsP(1:3), 0);
    dsP(4)   = min(max(dsP(4), 0.5), 3);
    dsP(5)   = min(max(dsP(5), 0.5), 3);
    dsP(6)   = min(max(dsP(6), -0.5), 0.5);
    dsP(7)   = max(dsP(7), 0);

    phiScale = 1;
    if strcmp(sampleType, 'phi_broad'); phiScale = cfg.phiBroadPhiScale; end
    tR = thetaRel*cos(dsP(6)) + phiRel*sin(dsP(6));
    pR = -thetaRel*sin(dsP(6)) + phiRel*cos(dsP(6));
    Y_lobe = dsP(7) * exp( ...
        -dsP(1)*max(tR,0).^dsP(4) ...
        -dsP(2)*max(-tR,0).^dsP(4) ...
        -dsP(3)*(phiScale*abs(pR)).^dsP(5));

    scores = zeros(1, model.bestK);
    for k = 1:model.bestK
        scores(k) = predict_local(model.scoreModels{k}, X);
    end

    Y_pred = max(Y_lobe(:).' + scores * model.basis' + model.muRes, 0);
    Emap = reshape(Y_pred, size(thetaRel));
end

function X = build_features_local(U, f, th, pol)
    pn = double(strcmpi(pol, 'TM'));
    X = [1, U, U^2, log10(f), th, th^2, cosd(th), pn, U*log10(f), U*th];
end

function yhat = predict_local(mdl, X)
    switch mdl.type
        case {'gpr','ensemble'}
            yhat = predict(mdl.model, X);
        case 'linear'
            yhat = X * mdl.Beta;
        otherwise
            yhat = 0;
    end
    yhat(isnan(yhat)|isinf(yhat)) = 0;
end