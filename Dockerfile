FROM thomasweise/docker-pandoc-r:latest

RUN Rscript -e "install.packages('bookdown')"