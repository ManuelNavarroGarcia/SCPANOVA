improved_install_packages<-function(packages_to_use) {
    # https://stackoverflow.com/a/44660688 
    libs<-unlist(list(packages_to_use))
    # Get a list of the packages and their requirement value (boolean)
    req<-suppressPackageStartupMessages(unlist(lapply(libs,require,character.only=TRUE)))
    # Get the non installed packages
    need<-libs[req==FALSE]
    if(length(need)>0){ 
        install.packages(need, repos='http://cran.us.r-project.org')
        lapply(need,require,character.only=TRUE)
    }
}