-- Keep the original LaTeX source while writing equivalent Word equations.
function Math(node)
  node.text = node.text:gsub("\\Bigl", "\\left"):gsub("\\Bigr", "\\right")
  node.text = node.text:gsub("Ref%.~\\cite%{psn2027%}", "the PSN paper")
  return node
end
