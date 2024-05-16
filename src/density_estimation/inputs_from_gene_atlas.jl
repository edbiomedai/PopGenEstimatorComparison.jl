########################################################################
###       Creation of DensityEstimation Inputs From geneATLAS        ###
########################################################################

trait_chr_associations_filename(trait, chr) = string("imputed.allWhites.", trait, ".chr", chr, ".csv.gz")

chr_filename(chr) = string("snps.imputed.chr", chr, ".csv.gz")

function download_gene_atlas_trait_info(trait; outdir="gene_atlas_data")
    isdir(outdir) || mkdir(outdir)
    # Download association results per chromosome
    for chr in 1:22
        filename = trait_chr_associations_filename(trait, chr) 
        url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/imputed/data.copy/", filename)
        download(url, joinpath(outdir, filename))
    end
end

"""
    download_gene_atlas_data(traits=; outdir="gene_atlas_data", trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))

Create an `outdir` folder and download summary statistics for each trait in `traits` in dedicated subfolders. 
Each subfolder will contain 22 files, one for each chromosome. The `trait_table_path` is used to map the trait name to the download key.
"""
function download_gene_atlas_data(traits, trait_key_map; outdir="gene_atlas_data")
    isdir(outdir) || mkdir(outdir)
    # Download traits association results in dedicated subfolders
    for trait in traits
        trait_key = trait_key_map[trait]
        trait_outdir = joinpath(outdir, replace(trait_key, " " => "_"))
        download_gene_atlas_trait_info(trait_key; outdir=trait_outdir)
    end
    # Download variants info
    variants_dir = joinpath(outdir, "variants_info")
    isdir(variants_dir) || mkdir(variants_dir)
    for chr in 1:22
        filename = chr_filename(chr)
        url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/snps/extended/", filename)
        download(url, joinpath(variants_dir, filename))
    end
end

function get_trait_to_variants_from_estimands(estimands; regex=r"^rs[0-9]*")
    trait_to_variants = Dict()
    for Ψ in estimands
        outcome = get_outcome(Ψ)
        variants = filter(x -> occursin(regex, string(x)), get_treatments(Ψ))
        if haskey(trait_to_variants, outcome) && !isempty(variants)
            union!(trait_to_variants[outcome], variants)
        else
            trait_to_variants[outcome] = Set(variants)
        end
    end
    return trait_to_variants
end

function update_trait_to_variants_from_gene_atlas!(trait_to_variants, trait_key_map; 
    gene_atlas_dir="gene_atlas_data",
    maf=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6
    )
    for trait in traits
        trait_key = trait_key_map[trait]
        estimand_variants = trait_to_variants[trait]
        independent_chr_variants = Dict()
        for chr in 1:22
            # Load association data and SNP info
            associations = CSV.read(joinpath(gene_atlas_dir, trait_key, PopGenEstimatorComparison.trait_chr_associations_filename(trait_key, chr)), 
                header=["SNP", "ALLELE", "NBETA", "NSE", "PV"],
                skipto=2,
                DataFrame
            )
            variants_info = CSV.read(joinpath(gene_atlas_dir, "variants_info", PopGenEstimatorComparison.chr_filename(chr)) , DataFrame)
            associations = innerjoin(associations, variants_info, on="SNP")
            associations.NOTIN_ESTIMAND = [v ∉ estimand_variants for v in associations.SNP]
            # Only keep variants in estimands or bi-allelic SNPs with "sufficient" MAF and p-value
            filter!(
                x -> !(x.NOTIN_ESTIMAND) || (x.PV < pvalue_threshold && x.MAF >= maf && (length(x.A1) == length(x.A2) == 1)), 
                associations
            )
            # Prioritizing SNPs in estimands and then by p-value
            sort!(associations, [:NOTIN_ESTIMAND, :PV])
            snp_to_pos = []
            for (snp, pos, notin_estimand) ∈ zip(associations.SNP, associations.Position, associations.NOTIN_ESTIMAND)
                # Always push SNPs in estimands
                if !notin_estimand
                    push!(snp_to_pos, snp => pos)
                else
                    # Always push the first SNP
                    if isempty(snp_to_pos)
                        push!(snp_to_pos, snp => pos)
                    else
                        # Only push if the SNP is at least `distance_threshold` away from the closest SNP
                        min_dist = min(abs(prev_pos - pos) for (prev_snp, prev_pos) in snp_to_pos)
                        if min_dist > distance_threshold
                            push!(snp_to_pos, snp => pos)
                        end
                    end
                end
            end

            # Update variant set from associated SNPs
            independent_chr_variants[chr] = [x[1] for x in snp_to_pos]
        end
        independent_variants = vcat(values(independent_chr_variants))
        # Check all variants in estimands have been found (i.e genotyped)
        notfound = setdiff(estimand_variants, independent_variants)
        isempty(notfound) || throw(ArgumentError(string("Did not find some estimands' variants in geneATLAS: ", notfound)))
        trait_to_variants[trait] = independent_variants
    end
end

"""
    get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))

Retrieve the geneAtlas key from the Description. This will fail if not all traits are present in the geneAtlas.
"""
function get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))
    trait_table = CSV.read(trait_table_path, DataFrame)
    trait_subset = filter(x -> x.Description ∈ traits, trait_table)
    not_found_traits = setdiff(traits, trait_subset.Description)
    @assert isempty(not_found_traits) || throw(ArgumentError(string("Did not find the following traits in the geneATLAS: ", not_found_traits)))
    return Dict(row.Description => row.key for row in eachrow(trait_subset))
end

function density_estimation_inputs_from_gene_atlas(estimands, genotypes_prefix;
    gene_atlas_dir="gene_atlas_data", 
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
    maf=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6
    )
    # Retrieve traits and variants from estimands
    trait_to_variants = get_trait_to_variants_from_estimands(estimands)
    # Retrive Trait to geneAtlas key map
    trait_key_map = get_trait_key_map(keys(trait_to_variants), trait_table_path=trait_table_path)
    # Retrieve traits and variant info from geneAtlas
    download_gene_atlas_data(traits, trait_key_map; outdir=gene_atlas_dir)
    # Update variant set for each trait using geneAtlas summary statistics
    update_trait_to_variants_from_gene_atlas!(trait_to_variants, trait_key_map; 
        gene_atlas_dir=gene_atlas_dir,
        maf=maf,
        pvalue_threshold=pvalue_threshold,
        distance_threshold=distance_threshold
    )
    return trait_to_variants
end