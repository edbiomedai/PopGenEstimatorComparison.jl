# This is NOT a reprodubile script, it is only informative.

using BGEN

chr_to_snp = Dict(
    21 => ["rs115397110"],
    10 => ["rs184270108", "rs4506565"],
    19 => ["rs10419224", "rs59103106", "rs8111699"],
    20 => ["rs6059655", "rs6088372"],
    15 => ["rs1129038", ],
    14 => ["rs10132320"],
    16 => ["rs1805005", "rs1805007", "rs1805008", "rs9926016", "rs9940128"],
    17 => ["rs11868112", "rs3859191"],
    3 => ["rs1732170", "rs62295911"],
    4 => ["rs356219"],
    2 => ["rs62107261", "rs9287850"],
    5 => ["rs456998", "rs974766"],
    6 => ["rs6456121", "rs3129716", "rs3129889", "rs502771", "rs9268219"]
)

snps_genotypes = Dict()

for (chr, snps) ∈ chr_to_snp
    bgen_prefix = string("/home/s2042526/UK-BioBank-53116/imputed/ukb_53116_chr", chr)
    bgen_file = string(bgen_prefix, ".bgen")
    idx_file = string(bgen_prefix, ".bgen.bgi")
    sample_file = string(bgen_prefix, ".sample")
    b = Bgen(bgen_file; sample_path=sample_file, idx_path=idx_file)
    for snp in snps
        v = variant_by_rsid(b, snp)
        alls = alleles(v)
        genotypes = [alls[1]*alls[1], alls[1]*alls[2], alls[2]*alls[2]]
        maf = mean(minor_allele_dosage!(b, v))/2
        snps_genotypes[snp] = (genotypes, maf)
    end
end