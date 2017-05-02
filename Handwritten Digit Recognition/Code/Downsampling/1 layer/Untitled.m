load('norm_imgs_train.mat','norm_imgs_train');
norm_imgs_train_down = downsample(norm_imgs_train,100);

load('norm_imgs_val.mat','norm_imgs_val');
norm_imgs_val_down = downsample(norm_imgs_val,100);

 load('norm_imgs_test.mat','norm_imgs_test');
 norm_imgs_test_down = downsample(norm_imgs_test,100);