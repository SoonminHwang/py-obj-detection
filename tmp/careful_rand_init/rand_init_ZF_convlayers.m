figure; 

%%
conv1 = imread( 'conv1_int16.tif' );
conv1 = single( conv1 ) ./ 256 ./256;

subplot(3,4,1); hold on;
hConv1 = plot( conv1(:), 'b.' );
hRand01 = plot( normrnd( 0, 0.1, [numel(conv1), 1]), 'm.' );
legend( [hConv1, hRand01], {'conv1', 'rand, sig=0.1'} );

%%
conv2 = imread( 'conv2_int16.tif' );
conv2 = single( conv2 ) ./ 256 ./256;

init = zeros( numel(conv2), 1 );
init( 1:100:end ) = normrnd( 0, 0.02, [ceil( numel(conv2) / 100), 1]);

subplot(3,4,2); hold on; cla;
hConv2 = plot( conv2(:), 'b.' );
hRand001 = plot( init, 'm.' );
legend( [hConv2, hRand001], {'conv2', 'rand, sig=0.01'} );

%%
conv3 = imread( 'conv3_int16.tif' );
conv3 = single( conv3 ) ./ 256 ./256;

init = zeros( numel(conv3), 1 );
init( 1:100:end ) = normrnd( 0, 0.02, [ceil( numel(conv3) / 100), 1]);

subplot(3,4,3); hold on; cla;
hConv3 = plot( conv3(:), 'b.' );
hRand002 = plot( init, 'm.' );
legend( [hConv3, hRand002], {'conv3', 'rand, sig=0.02'} );

%%
conv4 = imread( 'conv4_int16.tif' );
conv4 = single( conv4 ) ./ 256 ./256;

init = zeros( numel(conv4), 1 );
init( 1:100:end ) = normrnd( 0, 0.02, [ceil( numel(conv4) / 100), 1]);

subplot(3,4,4); hold on; cla;
hConv4 = plot( conv4(:), 'b.' );
hRand002 = plot( init, 'm.' );
legend( [hConv4, hRand002], {'conv4', 'rand, sig=0.02'} );

%%
conv5 = imread( 'conv5_int16.tif' );
conv5 = single( conv5 ) ./ 256 ./256;

init = zeros( numel(conv5), 1 );
init( 1:100:end ) = normrnd( 0, 0.02, [ceil( numel(conv5) / 100), 1]);

subplot(3,4,5); hold on;    cla;
hConv5 = plot( conv5(:), 'b.' );
hRand002 = plot( init, 'm.' );
legend( [hConv5, hRand002], {'conv5', 'rand, sig=0.02'} );

%%
fc6 = imread( 'fc6_int16.tif' );
fc6 = single( fc6 ) ./ 256 ./256;

init = zeros( numel(fc6), 1 );
init( 1:10000:end ) = normrnd( 0, 0.01, [ceil( numel(fc6) / 10000), 1]);

subplot(3,4,6); hold on;    cla;
hFc6 = plot( fc6(:), 'b.' );
hRand001 = plot( init, 'm.' );
legend( [hFc6, hRand001], {'fc6', 'rand, sig=0.01'} );

%%
fc7 = imread( 'fc7_int16.tif' );
fc7 = single( fc7 ) ./ 256 ./256;

init = zeros( numel(fc7), 1 );
init( 1:1000:end ) = normrnd( 0, 0.01, [ceil( numel(fc7) / 1000), 1]);

subplot(3,4,7); hold on;    cla;
hFc7 = plot( fc7(:), 'b.' );
hRand001 = plot( init, 'm.' );
legend( [hFc7, hRand001], {'fc7', 'rand, sig=0.01'} );

%%
rpn_conv_3x3 = imread( 'rpn_conv_3x3_int16.tif' );
rpn_conv_3x3 = single( rpn_conv_3x3 ) ./ 256 ./256;

init = zeros( numel(rpn_conv_3x3), 1 );
init( 1:100:end ) = normrnd( 0, 0.01, [ceil( numel(rpn_conv_3x3) / 100), 1]);

subplot(3,4,8); hold on;    cla;
hRpnConv = plot( rpn_conv_3x3(:), 'b.' );
hRand001 = plot( init, 'm.' );
legend( [hRpnConv, hRand001], {'rpn_conv_3x3', 'rand, sig=0.01'} );

%%
rpn_cls_score = imread( 'rpn_cls_score_int16.tif' );
rpn_cls_score = single( rpn_cls_score ) ./ 256 ./256;

init = zeros( numel(rpn_cls_score), 1 );
init( 1:10:end ) = normrnd( 0, 0.02, [ceil( numel(rpn_cls_score) / 10), 1]);

subplot(3,4,9); hold on;    cla;
hRpnCls = plot( rpn_cls_score(:), 'b.' );
hRand002 = plot( init, 'm.' );
legend( [hRpnCls, hRand002], {'rpn_cls_score', 'rand, sig=0.02'} );



%%
rpn_bbox_pred = imread( 'rpn_bbox_pred_int16.tif' );
rpn_bbox_pred = single( rpn_bbox_pred ) ./ 256 ./256;

init = zeros( numel(rpn_bbox_pred), 1 );
init( 1:10:end ) = normrnd( 0, 0.01, [ceil( numel(rpn_bbox_pred) / 10), 1]);

subplot(3,4,10); hold on;    cla;
hRpnBbox = plot( rpn_bbox_pred(:), 'b.' );
hRand001 = plot( init, 'm.' );
legend( [hRpnBbox, hRand001], {'rpn_bbox_pred', 'rand, sig=0.01'} );



