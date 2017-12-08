
#filter missing data columns
missing_features = get_features_missing_data(house_train, 0)
filter_features(house_train, missing_features)
house_train.shape
house_train.info()


#smooth the sale price using log transformation(smoothening outlier data)
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
features = ['SalePrice','log_sale_price']
viz_cont(house_train, features)

#explore relationship of neighborhood to saleprice
target = 'SalePrice'
features = ['Neighborhood']
viz_cat_cont_box(house_train, features, target)

#explore relationship of livarea and totalbsmt to saleprice
features = ['GrLivArea','TotalBsmtSF']
viz_cont_cont(house_train, features, target)

filter_features(house_train, ['Id'])
                               
#explore relation among all continuous features vs saleprice 
corr = get_heat_map_corr(house_train)
get_target_corr(corr, 'SalePrice')
get_target_corr(corr, 'log_sale_price')

#do one-hot-encoding for all the categorical features
print(get_categorical_columns(house_train))
house_train1 = one_hot_encode(house_train)
house_train1.shape
house_train1.info()

filter_features(house_train1, ['SalePrice','log_sale_price'])
X_train = house_train1
y_train = house_train['log_sale_price']

rf_estimator = ensemble.RandomForestRegressor(random_state=2017)
rf_grid = {'max_features':[9,10,11,12,15,16], 'n_estimators':[50, 100,200]}
model = fit_model(rf_estimator, rf_grid, X_train, y_train)