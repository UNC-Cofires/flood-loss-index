# Overview

This folder contains code to stochastically assign anonymized National Flood Insurance Program (NFIP) claim and policy records to buildings based on available geographic attributes.

## Data sources

### Insurance data

Data on historical NFIP claims and policies were obtained from the [FIMA NFIP Redacted Claims v2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) and [FIMA NFIP Redacted Policies v2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2) datasets available through [OpenFEMA](https://www.fema.gov/about/reports-and-data/openfema). These datasets are anonymized to protect the identity of policyholders, but contain information on geographic identifiers (e.g., zip code, census block group, flood zone, etc.) that can be used to link records to specific neighborhoods and collections of buildings. These datasets include the full history of NFIP claims since 1978 and policy records since 2009.

### Building data

Data on building locations and attributes were obtained from the 2022 version of the [National Structure Inventory (NSI)](https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation) base layer maintained by the U.S. Army Corps of Engineers (USACE). This dataset provides information on the locations, occupancy types (e.g., residential single-family), and basic structural attributes (e.g., foundation type, number of floors) of 123 million buildings across the United States. Although this nationwide structure inventory is a valuable resource for mapping the location of buildings at scale, there are several [known issues](https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/frequently-asked-questions#id-.FrequentlyAskedQuestionsv2022-KnownIssues) with the dataset that users should keep in mind. Notably, certain structure attributes (e.g., foundation types) are often stochastically generated based on region-specific distributions, and may differ from higher-quality local datasets. Because we are mainly interested in the location and occupancy type of buildings, we do not expect these issues to have a substantial impact on our analysis. 

### Geographic data

Information on the following geographic identifiers was spatially joined to the NSI building points to facilitate comparisons with NFIP data. 

| Variable                | Description                                 | Source                                                                                                                                      |
|-------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 2020 ZCTA               | ZIP code tabulation area in the 2020 census | [2020 TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2020.html#list-tab-790442341) |
| 2000 Census block group | Census block group GEOID in the 2000 census | [2000 TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2000.html#list-tab-790442341) |
| 2010 Census block group | Census block group GEOID in the 2010 census | [2010 TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2010.html#list-tab-790442341) |
| 2020 Census block group | Census block group GEOID in the 2020 census | [2020 TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2020.html#list-tab-790442341) |
| 2025 Flood zone         | Flood zone as of January 2025               | [FEMA National Flood Hazard Layer](https://hazards.fema.gov/femaportal/NFHL/searchResult/)                                                  |


## Identifying potential matches between NFIP records and building points

The `match_claims_to_buildings.py` and `match_policies_to_buildings.py` scripts are used to identify potential matches between NFIP claim and policy records and NSI building points based on combinations of geographic attributes available in both datasets. By considering multiple attributes in combination, it is possible to locate each NFIP record to a smaller group of buildings than would be possible using each attribute in isolation. For each NFIP record, matching buildings are identified based on the following attributes:

| Priority | Variable                              | Data type | Column name*                     | Multiple values* | Notes                                                                                                                                                               |
|----------|---------------------------------------|-----------|---------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1        | Rounded latitude                    | string    | `match_latitude`                  | N               | Cleaned version of `latitude` column in OpenFEMA. Rounded to nearest 0.1 degree.                                                                                           |
| 2        | Rounded longitude                   | string    | `match_longitude`                 | N               | Cleaned version of `longitude` column in OpenFEMA. Rounded to nearest 0.1 degree.                                                                                  |
| 3        | Census block group              | string    | `match_censusBlockGroupFips`*      | Y               | Cleaned version of `censusBlockGroupFips` column in OpenFEMA. Can take on multiple values in buildings data (see footnote).                                                                                                       |
| 4        | SFHA flag | integer   | `match_sfhaIndicator`             | N               | Derived from `ratedFloodZone` column in OpenFEMA.                                                                                                                   |
| 5        | Coastal flood zone flag               | integer   | `match_coastalFloodZoneIndicator` | N               | Derived from `ratedFloodZone` column in OpenFEMA.                                                                                                                   |
| 6        | Zip code                     | string    | `match_reportedZipCode`           | N               | Cleaned version of `reportedZipCode` column in OpenFEMA. Used 2025 ZCTA as a proxy for zip code in buildings data.                                                                                                           |
| 7        | Broad occupancy type                  | string    | `match_simplifiedOccupancyType`   | N               | Derived from `occupancyType` column in OpenFEMA. Occupancy codes were collapsed into four categories: `residential_single_family`,`residential_two_to_four_family`,`residential_other`, and `non_residential`. |

*Because OpenFEMA does not record the vintage year associated with census block group GEOIDs, we allow NFIP records to match to any building with a matching GEOID in any of the 2000, 2010, or 2020 vintage years. To facilitate this, the buildings dataset should include a column called `match_censusBlockGroupFips_values` where the GEOID of each building across different vintage years is recorded as a comma-separated list (e.g.,`"['440010301001','440070107021','440070107023']"`).

We used the following algorithm: 

![alt text](https://github.com/UNC-Cofires/flood-loss-index/blob/main/nfip_building_matching/images/building_matching_flowchart.png "Logo Title Text 1")




## Generating presence-absence points for specific events