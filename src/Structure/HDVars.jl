struct HVars{Aphys, Atrans, usr_var} <: MHDVars
  "x-component of velocity"
    ux :: Aphys
  "y-component of velocity"
    uy :: Aphys
  "z-component of velocity"
    uz :: Aphys

  # Temperatory Cache 
  "Non-linear term 1"
   nonlin1 :: Aphys
  "Fourier transform of Non-linear term"
   nonlinh1 :: Atrans

  # User Defined Vars
  "User Defined Vars"
    usr_vars :: usr_var
end

struct CHVars{Aphys, Atrans, usr_var} <: MHDVars
  "density "
    ρ  :: Aphys
  "x-component of velocity"
    ux :: Aphys
  "y-component of velocity"
    uy :: Aphys
  "z-component of velocity"
    uz :: Aphys

  "x-component of fourier velocity"
   uxh :: Atrans
  "y-component of fourier velocity"
   uyh :: Atrans
  "z-component of fourier velocity"
   uzh :: Atrans

  # Temperatory Cache 
  "Non-linear term 1"
   nonlin1 :: Aphys
  "Fourier transform of Non-linear term"
   nonlinh1 :: Atrans

  "Non-linear term 2"
   nonlin2 :: Aphys
  "Fourier transform of Non-linear term"
   nonlinh2 :: Atrans

  # User Defined Vars
  "User Defined Vars"
    usr_vars :: usr_var
end
