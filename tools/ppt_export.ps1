$log = "E:\Remote\项目\AMR\tools\ppt_export.log"
"start" | Out-File $log -Encoding utf8
try {
  $src = "E:\Remote\项目\AMR\docs\答辩\答辩PPT.pptx"
  $pdf = "E:\Remote\项目\AMR\docs\答辩\答辩PPT.pdf"
  $ppt = New-Object -ComObject PowerPoint.Application
  "app ok" | Out-File $log -Append -Encoding utf8
  $pres = $ppt.Presentations.Open($src, $true, $false, $false)
  "opened, slides=" + $pres.Slides.Count | Out-File $log -Append -Encoding utf8
  $pres.SaveAs($pdf, 32)
  "pdf ok" | Out-File $log -Append -Encoding utf8
  $pres.Close()
  $ppt.Quit()
  "done" | Out-File $log -Append -Encoding utf8
} catch {
  "ERR: " + $_.Exception.Message | Out-File $log -Append -Encoding utf8
  try { $ppt.Quit() } catch {}
}