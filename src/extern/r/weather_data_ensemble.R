# script to generate a csv with weather data.

# start with weekly weather data for a place for many years.

station = getStationCode("Boston")
stationcode = station[1][[1]][1,3]

checkDataAvailabilityForDateRange(stationcode,
                                  "1973-01-01",
                                  "2013-12-31")

tempdata = getWeatherForDate("SEA", "2014-04-01", end_date="2015-03-31", opt_write_to_file=FALSE)

drops <- c("Date")
tempdata <- tempdata[ , !(names(tempdata) %in% drops)]

tempdata_mat <- data.matrix(tempdata)

write.table(tempdata_mat, file = "year.csv", sep = ",", col.names = FALSE, qmethod = "double", row.names=FALSE)


# main loop starts here

for (year in 1973:1977){
  startdate <- paste(c(year, "-04-01"), collapse = "")
  enddate <- paste(c(year+1, "-03-31"), collapse = "")
  filename <- paste(c(year, ".csv"), collapse = "")
  print(paste(startdate,enddate))
  tempdata = getWeatherForDate("SEA", startdate, end_date=enddate, opt_write_to_file=FALSE)
  drops <- c("Date")
  tempdata <- tempdata[ , !(names(tempdata) %in% drops)]
  tempdata_mat <- data.matrix(tempdata)
  write.table(tempdata_mat, file = filename, sep = ",", col.names = FALSE, qmethod = "double", row.names=FALSE)
}