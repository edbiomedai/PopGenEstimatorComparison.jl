function multivariants_estimands_groups(estimands)

end

function gwas_based_density_estimation_inputs(datasetfile, estimands_prefix; batchsize=10, output_prefix="de_")
    estimands_dir, _prefix = splitdir(estimands_prefix)
    _estimands_dir = estimands_dir == "" ? "." : estimands_dir

    estimand_files = map(
        f -> joinpath(estimands_dir, f),
        filter(
            f -> startswith(f, _prefix), 
            readdir(_estimands_dir)
        )
    )
    dataset = TargetedEstimation.instantiate_dataset(datasetfile)
    estimands = reduce(
        vcat,
        TargetedEstimation.instantiate_estimands(f, dataset) for f in estimand_files
    )
    estimand_groups = make_compatible_estimands_groups(estimands)
    for (group_index, group) in enumerate(estimand_groups)
        group_prefix = string(output_prefix, "group_", group_index)
        # Write estimands
        for (batch_index, batch) in enumerate(Iterators.partition(group.estimands, batchsize))
            batch_filename = string(group_prefix, "_estimands_", batch_index, ".jls")
            serialize(batch_filename, TMLE.Configuration(estimands=batch))
        end
        # Write conditional densities
        for (cd_index, (outcome, parents)) in enumerate(group.conditional_densities)
            conditional_density_filename = string(group_prefix, "_conditional_density_", cd_index, ".json")
            open(conditional_density_filename, "w") do io
                JSON.print(io, Dict("outcome" => outcome, "parents" => parents), 1)
            end
        end
    end
end

########################################################################
### Creation of Estimands & Conditional Distributions From geneATLAS ###
########################################################################

"""
This procedure produces a conditional distribution for each trait corresponding to some of the `files_prefix`.
Eventually, the conditional distribution is of the form ``P(Y| Variants, PCs, Covariates)``. Since ``PCs`` and 
``Covariates`` are fixed, only variants need to be determined. These are chosen based on a previous association study 
(geneATLAS) and only independent (`rsquared_threshold`) and associated (`pvalue_threshold`) variants are kept.

# Arguments

- files_prefix: A set of trait-based files downloaded from the geneATLAS. For each trait, there should be as many files as the number of chromosomes.
- pvalue_threshold: Only variant passing this significance threshold are used in the conditional density.
- rsquared_threshold: Only variants that are not too correlated with each other are kept.

# Returns

A set of conditional distributions
"""
function conditional_distributions_from_geneATLAS(files_prefix;
    pvalue_threshold=1e-6, 
    rsquared_threshold=0.5
    )
    # Group files per trait and/or chromosome
    trait_to_chr_files = Dict()
    chr_files = Dict()
    for file in files_matching_prefix(files_prefix)
        basefilename = basename(file)
        file_pieces = split(basefilename, ".")
        chr = file_pieces[end-1]
        ## If the file is a SNP info file
        if startswith(basefilename, "snps")
            chr_files[chr] = file
        ## Else it is a GWAS file
        else
            trait = file_pieces[end-2]
            if haskey(trait_to_chr_files, trait)
                trait_to_chr_files[trait][chr] = file
            else
                trait_to_chr_files[trait] = Dict(chr => file)
            end
        end
    end
    # Find trait associated variants matching criteria
    traits_to_variants = Dict()
    for (trait, chr_dict) in trait_to_chr_files
        traits_to_variants[trait] = []
        for (chr, gwas_file) ∈ chr_dict
            snps_file = chr_files[chr]
            chr_variants = traits_associated_variants(gwas_file, snps_file; 
                pvalue_threshold=pvalue_threshold, 
                rsquared_threshold=rsquared_threshold
            )
            append!(traits_to_variants[trait], chr_variants)
        end
    end
end

function traits_associated_variants(
    gwas_file="/Users/olivierlabayle/Downloads/genotyped.allWhites.clinical_c_Block_J40-J47.chr1.csv",
    snps_file="/Users/olivierlabayle/Downloads/snps.genotyped.chr1.csv";
    pvalue_threshold = 1e-6,
    rsquared_threshold = 0.5
    )
    gwas_results = CSV.read(gwas_file, DataFrame;
        header=["SNP", "ALLELE", "NBETA", "NSE", "PV"],
        skipto=2
    )
    # Only keep variant that are associated with the trait
    filter!(x -> x.PV < pvalue_threshold, gwas_results)
    # Only keep variants that are not correlated
    snps = CSV.read(snps_file, DataFrame)
    results = innerjoin(gwas_results, snps, on="SNP")
    ## Ensure most significantly associated are kept first
    sort!(results, :PV)
    ## Get the non-correlated set
    variant_set = [results.SNP[1]]
    for new_variant in results.SNP[2:end]
        correlated = false
        for validated_variant in variant_set
            snp_data = dropmissing(dataset[!, [new_variant, validated_variant]])
            rsquared = corr(snp_data[!, new_variant], snp_data[!, validated_variant])^2
            if rsquared > rsquared_threshold
                # Stop fast
                correlated = true
                break
            end
        end
        if correlated === false
            push!(variant_set, new_variant)
        end
    end
    
    return variant_set
end


function download_gene_atlas_trait_info(trait; outdir="gene_atlas_data")
    isdir(outdir) || mkdir(outdir)
    # Download association results per chromosome
    for chr in 1:22
        filename = string("genotyped.allWhites.", trait, ".chr", chr, ".csv.gz")
        url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/genotyped/data/", filename)
        download(url, joinpath(outdir, filename))
    end
end

"""
    download_gene_atlas_data(traits=; outdir="gene_atlas_data", trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))

Create an `outdir` folder and download summary statistics for each trait in `traits` in dedicated subfolders. 
Each subfolder will contain 22 files, one for each chromosome. The `trait_table_path` is used to map the trait name to the download key.
"""
function download_gene_atlas_data(traits; 
    outdir="gene_atlas_data", 
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv")
    )
    isdir(outdir) || mkdir(outdir)
    # Download traits association results in dedicated subfolders
    for trait in traits
        trait_key = trait_table[only(findall(==(trait), trait_table.Description)), :key]
        trait_outdir = joinpath(outdir, replace(trait_key, " " => "_"))
        download_gene_atlas_trait_info(trait_key; outdir=trait_outdir)
    end
end

function get_trait_to_variants_from_estimands(estimands; regex=r"^rs[0-9]*")
    trait_to_variants = Dict()
    for Ψ in estimands
        outcome = get_outcome(Ψ)
        variants = filter(x -> occursin(regex, string(x)), get_treatments(Ψ))
        isempty(variants) && continue
        if haskey(trait_to_variants, outcome)
            union!(trait_to_variants[outcome], variants)
        else
            trait_to_variants[outcome] = Set(variants)
        end
    end
    return trait_to_variants
end

function update_trait_to_variants_from_gene_atlas!(traits, genotypes_prefix;outdir="gene_atlas_data")
    for trait in traits
        trait_key = trait_table[only(findall(==(trait), trait_table.Description)), :key]

    end
end

function conditional_distributions_from_gene_atlas(estimands, genotypes_prefix;
    outdir="gene_atlas_data", 
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv")
    )
    # Retrieve traits and variants from estimands
    trait_to_variants = get_trait_to_variants_from_estimands(estimands)
    trait_table = CSV.read(trait_table_path, DataFrame)
    trait_key = trait_table[only(findall(∈(keys(traits)), trait_table.Description)), :key]

    # Retrieve traits info from geneAtlas
    download_gene_atlas_data(traits; outdir=outdir, trait_table_path=trait_table_path)

    # Update variant set for each trait using geneAtlas summary statistics
    update_trait_to_variants_from_gene_atlas!(traits, genotypes_prefix;outdir=outdir)
end