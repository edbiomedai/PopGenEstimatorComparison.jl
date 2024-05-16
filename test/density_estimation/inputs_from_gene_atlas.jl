module TestInputsFromGeneAtlas

using TMLE
using Test
using PopGenEstimatorComparison

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

@testset "Test get_trait_key_map"
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
    copy!(ARGS, [
        "density-estimation-inputs-from-ga",
        "estimands_file"
        datasetfile,
        estimands_prefix,
        string("--output-prefix=", output_prefix),
        string("--batchsize=10")
    ])
    PopGenEstimatorComparison.julia_main()

    density_estimation_inputs_from_gene_atlas(estimands, genotypes_prefix;
        gene_atlas_dir="gene_atlas_data", 
        trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
        maf=0.01,
        pvalue_threshold=1e-5,
        distance_threshold=1e6
    )
end

end

true