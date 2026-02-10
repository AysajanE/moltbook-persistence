# Repo-local latexmk configuration.
#
# When latexmk is invoked from the repository root with a path like
#   latexmk -pdf paper/main_*.tex
# LaTeX would otherwise write build artifacts (e.g., .aux/.log/.fls) into the
# current working directory. Enabling -cd behavior keeps artifacts next to the
# corresponding source file (e.g., under paper/).
$do_cd = 1;
