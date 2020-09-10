# Sadedegel Static Docs

This subdirectory for the Sadedegel repository contains all of its documentation.

## Installation

```bash
# install Ruby

brew install ruby
echo 'export PATH="/usr/local/opt/ruby/bin:$PATH"' >> ~/.bash_profile
which ruby

# install Jekyll

gem install --user-install bundler jekyll
# check the Ruby version
ruby -v
# Append your path file with the following, replacing the X.X with the first two digits of your Ruby version
echo 'export PATH="$HOME/.gem/ruby/X.X.0/bin:$PATH"' >> ~/.bash_profile

# install Bundler

gem install bundler
bundle install
```

## Run

```bash
bundle exec jekyll serve -b /sadedegel
```
