using Documenter

makedocs(;
         repo = "https://github.com/sebapersson/petab_sciml/blob/{commit}{path}#{line}",
         checkdocs = :exports,
         warnonly = false,
         sitename = "PEtab SciML extension",
         format = Documenter.HTML(;prettyurls = get(ENV, "CI", "false") == "true",
                                   repolink = "https://github.com/sebapersson/petab_sciml",
                                   edit_link = "main"),
         pages = [
             "Home" => "index.md",
             "Tutorial" => "tutorial.md",
             "Format" => "format.md",
             "Network Layers and Activation Functions" => "net_activation.md"],)
