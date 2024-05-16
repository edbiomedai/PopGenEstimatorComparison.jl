module TestInputsFromGeneAtlas

using TMLE
using Test
using PopGenEstimatorComparison
using Serialization

function test_estimands()
    return [
        factorialEstimand(ATE, (rs502771=["TT", "TC", "CC"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(ATE, (rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(IATE, (rs502771=["TT", "TC", "CC"], rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(IATE, (rs11868112=["CC", "CT", "TT"], rs6456121=["CC", "CT", "TT"], rs356219=["GG", "GA", "AA"]), "G20 Parkinson's disease";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
    ]
end

function save_test_estimands(outdir)
    estimands = test_estimands()
    serialize(joinpath(outdir, "estimands_1.jls"), TMLE.Configuration(estimands=estimands[1:2]))
    serialize(joinpath(outdir, "estimands_2.jls"), TMLE.Configuration(estimands=estimands[3:end]))
end

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test get_trait_to_variants_from_estimands" begin
    estimands = linear_interaction_dataset_ATEs().estimands
    # Empty regex
    trait_to_variants = PopGenEstimatorComparison.get_trait_to_variants_from_estimands(estimands; regex=r"")
    @test trait_to_variants == Dict(
        :Ycont  => Set([:T₁]),
        :Ybin   => Set([:T₁, :T₂]),
        :Ycount => Set([:T₁])
        )
    # T₂ regex
    trait_to_variants = PopGenEstimatorComparison.get_trait_to_variants_from_estimands(estimands; regex=r"T₂")
    @test trait_to_variants == Dict(
        :Ycont  => Set(),
        :Ybin   => Set([:T₂]),
        :Ycount => Set()
        )
    # T₁ regex
    trait_to_variants = PopGenEstimatorComparison.get_trait_to_variants_from_estimands(estimands; regex=r"T₁")
    @test trait_to_variants == Dict(
        :Ycont  => Set([:T₁]),
        :Ybin   => Set([:T₁]),
        :Ycount => Set([:T₁])
        )
end

@testset "Test get_trait_key_map" begin
    # "Vitamin D Level" and "Red-Hair" are not part of the geneATLAS
    traits = ["G35 Multiple sclerosis", "Vitamin D Level", "White blood cell (leukocyte) count", "sarcoidosis", "D86 Sarcoidosis", "G35 Multiple sclerosis", "K90-K93 Other diseases of the digestive system", "H00-H06 Disorders of eyelid, lacrimal system and orbit", "Trunk fat percentage"]
    @test_throws ArgumentError PopGenEstimatorComparison.get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))
    # Only valid traits
    traits = ["G35 Multiple sclerosis", "White blood cell (leukocyte) count", "sarcoidosis", "D86 Sarcoidosis", "G35 Multiple sclerosis", "K90-K93 Other diseases of the digestive system", "H00-H06 Disorders of eyelid, lacrimal system and orbit", "Trunk fat percentage"]
    trait_key_map = PopGenEstimatorComparison.get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))
    @test trait_key_map == Dict(
        "G35 Multiple sclerosis"                                 => "clinical_c_G35",
        "sarcoidosis"                                            => "selfReported_n_1414",
        "H00-H06 Disorders of eyelid, lacrimal system and orbit" => "clinical_c_Block_H00-H06",
        "Trunk fat percentage"                                   => "23127-0.0",
        "K90-K93 Other diseases of the digestive system"         => "clinical_c_Block_K90-K93",
        "D86 Sarcoidosis"                                        => "clinical_c_D86",
        "White blood cell (leukocyte) count"                     => "30000-0.0"
    )
end

@testset "Test density_estimation_inputs_from_gene_atlas" begin
    tmpdir = mktempdir()
    save_test_estimands(tmpdir)
    copy!(ARGS, [
        "simulation-inputs-from-ga",
        joinpath(tmpdir, "estimands"),
        string("--ga-download-dir=", joinpath(tmpdir, "gene_atlas_data")),
        "--remove-ga-data=true",
        string("--ga-trait-table=", joinpath("assets", "Traits_Table_GeneATLAS.csv")),
        "--maf-threshold=0.01",
        "--pvalue-threshold=1e-5",
        "--distance-threshold=1e6"
    ])
    PopGenEstimatorComparison.julia_main()

end

end

true