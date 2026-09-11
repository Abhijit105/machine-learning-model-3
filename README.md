# Dataset Overview (house_prices_ml_ready.csv)

- ## Total Records:

  10,000 clean records formatted for regression models (e.g., Random Forest Regressor, XGBoost, Linear Regression).

- ## Features Included:
  1. bedrooms: Number of bedrooms (discrete integers from 1 to 6).

  2. square_footage: Total floor area in square feet (continuous, scaled appropriately relative to bedroom count).

  3. bathrooms: Number of bathrooms (numerical with 0.5 increments).

  4. location: Categorical property location (Downtown, Suburban, Urban, Rural) ready for one-hot encoding or label encoding.

  5. age_of_house: Property age in years (continuous, 0 to 50).

  6. garage_size: Car capacity of the garage (discrete integers from 0 to 3).

  7. lot_size: Total lot area in square feet (continuous).

  8. distance_to_school: Distance to the nearest school in miles (continuous).

- ## Target Output Column:
  1. house_price: Market valuation in currency units, calculated using a realistic pricing function incorporating property features, location weighting coefficients, and controlled random noise.
