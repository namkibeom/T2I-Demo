# A Comparative Study of Text-to-Image Generation Models

## 1) minDALL-E , 2) GLIDE, 3) Stable Diffusion v1,v2, 4) Karlo

## Examples
<table>
  <table>
  <tr>
    <table> <tr>  Caption: Three people standing next to an elephant along a river </tr> <tr> <img src="image/MScoco_1.png" width="100" height="100"> </tr> </table>
  </tr>
  </table>
  <table>
  <tr>
    <td> 
        <table> <tr> minDALL-E </tr> <tr> <img src="image/minDALL-E_1.png" width="100" height="100"> </tr> </table>
    </td>
    <td width="220"> 
        <table> <tr> GLIDE </tr> <tr> <img src="image/GLIDE_1.png" width="100" height="100"> </tr> </table>
    </td>
    <td width="220"> 
       <table> <tr> Stable Diffusion v1 </tr> <tr> <img src="image/SDv1-4_1.png" width="100" height="100"> </tr> </table>
    </td>
    <td width="220"> 
        <table> <tr> Stable Diffusion v2 </tr> <tr> <img src="image/SDv2-1_1.png" width="100" height="100"> </tr> </table>
    </td>
    <td width="200">
      <table> <tr> Karlo </tr> <tr> <img src="image/karlo_1.png" width="100" height="100"> </tr> </table>
    </td>
  </tr>
  </table>
</table>

## Evaluation (Categorical Prompt)
| Model      | Total Params | Resolution   |      Time       |  Categorical-phr. | Categorical-SE |
|:----------:|:------------:|:------------:|:---------------:|:-----------------:|:--------------:|
| minDALL-E  |     1.3B     |  256 x 256   |   5.07 ± 0.035  |        0.75       |      0.74      |
| GLIDE      |     941M     |  256 x 256   |   5.64 ± 0.014  |        0.74       |      0.73      | 
| SD v1-4    |    859.52M   |  512 x 512   |   4.97 ± 0.050  |        0.82       |      0.80      | 
| SD v2-1    |    865.91M   |  512 x 512   |   3.77 ± 0.057  |        0.82       |      0.81      |
| Karlo      |     3.3B     |  256 x 256   |   5.05 ± 0.032  |        0.83       |      0.84      |


## Evaluation (By Category)
| Model      |  Color-adj.  | Color-n. |  Count+Pos-phr. | Count+Pos-SE |
|:----------:|:------------:|:--------:|:---------------:|:------------:|
| minDALL-E  |     0.81     |   0.81   |      0.74       |     0.73     |
| GLIDE      |     0.82     |   0.79   |      0.71       |     0.71     |
| SD v1-4    |     0.80     |   0.81   |      0.81       |     0.80     |
| SD v2-1    |     0.82     |   0.80   |      0.81       |     0.80     |
| Karlo      |     0.81     |   0.84   |      0.83       |     0.84     |



## Evaluation (MScoco-val14 / FID and CLIP score)
| Model      | zero-shot FID | CLIP Score  | 
|:----------:|:------------:|:------------:|
| minDALL-E  |     94.95    |    0.7248    |
| GLIDE      |     59.70    |    0.7011    |
| SD v1-4    |     47.49    |    0.8175    |
| SD v2-1    |     47.11    |    0.8308    |
| Karlo      |     43.59    |    0.8249    |





### minDALL-E
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@article{kakaobrain2021minDALL-E,
 title         = {minDALL-E on Conceptual Captions},
  author        = {Saehoon Kim, Sanghun Cho, Chiheon Kim, Doyup Lee, and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/minDALL-E}},
}"><pre class="notranslate"><code>@article{kakaobrain2021minDALL-,
  title         = {minDALL-E on Conceptual Captions},
  author        = {Saehoon Kim, Sanghun Cho, Chiheon Kim, Doyup Lee, and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/minDALL-E}},
}
</code></pre></div>

### GLIDE
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@article{glide,
      title         = { GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.},
      author        = {Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen},
      year          = {2021},
      howpublished  = {\url{[https://github.com/kakaobrain/minDALL-E](https://github.com/openai/glide-text2im)}},
}
}"><pre class="notranslate"><code>@article{glide,
      title         = { GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.},
      author        = {Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen},
      year          = {2021},
      howpublished  = {\url{[https://github.com/kakaobrain/minDALL-E](https://github.com/openai/glide-text2im)}},
}
</code></pre></div>

### Stable Diffusion
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="@article{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"><pre class="notranslate"><code>@article{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
</code></pre></div>
